import torch
from transformers import BertTokenizer
from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel
from torch import nn
from itertools import chain
from torch.nn import MSELoss, CrossEntropyLoss
from cleantext import clean
from num2words import num2words
import re
import string

punct_chars = list((set(string.punctuation) | {'’', '‘', '–', '—', '~', '|', '“', '”', '…', "'", "`", '_'}))
punct_chars.sort()
punctuation = ''.join(punct_chars)
replace = re.compile('[%s]' % re.escape(punctuation))

def get_num_words(text):
    if not isinstance(text, str):
        print("%s is not a string" % text)
    text = replace.sub(' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    text = re.sub(r'\[.+\]', " ", text)
    return len(text.split())

def number_to_words(num):
    try:
        return num2words(re.sub(",", "", num))
    except:
        return num


clean_str = lambda s: clean(s,
                            fix_unicode=True,  # fix various unicode errors
                            to_ascii=True,  # transliterate to closest ASCII representation
                            lower=True,  # lowercase text
                            no_line_breaks=True,  # fully strip line breaks as opposed to only normalizing them
                            no_urls=True,  # replace all URLs with a special token
                            no_emails=True,  # replace all email addresses with a special token
                            no_phone_numbers=True,  # replace all phone numbers with a special token
                            no_numbers=True,  # replace all numbers with a special token
                            no_digits=False,  # replace all digits with a special token
                            no_currency_symbols=False,  # replace all currency symbols with a special token
                            no_punct=False,  # fully remove punctuation
                            replace_with_url="<URL>",
                            replace_with_email="<EMAIL>",
                            replace_with_phone_number="<PHONE>",
                            replace_with_number=lambda m: number_to_words(m.group()),
                            replace_with_digit="0",
                            replace_with_currency_symbol="<CUR>",
                            lang="en"
                            )

clean_str_nopunct = lambda s: clean(s,
                            fix_unicode=True,  # fix various unicode errors
                            to_ascii=True,  # transliterate to closest ASCII representation
                            lower=True,  # lowercase text
                            no_line_breaks=True,  # fully strip line breaks as opposed to only normalizing them
                            no_urls=True,  # replace all URLs with a special token
                            no_emails=True,  # replace all email addresses with a special token
                            no_phone_numbers=True,  # replace all phone numbers with a special token
                            no_numbers=True,  # replace all numbers with a special token
                            no_digits=False,  # replace all digits with a special token
                            no_currency_symbols=False,  # replace all currency symbols with a special token
                            no_punct=True,  # fully remove punctuation
                            replace_with_url="<URL>",
                            replace_with_email="<EMAIL>",
                            replace_with_phone_number="<PHONE>",
                            replace_with_number=lambda m: number_to_words(m.group()),
                            replace_with_digit="0",
                            replace_with_currency_symbol="<CUR>",
                            lang="en"
                            )

def _get_clean_text(text, remove_punct=False):
    if remove_punct:
        return clean_str_nopunct(text)
    return clean_str(text)

class MultiHeadModel(BertPreTrainedModel):
  """Pre-trained BERT model that uses our loss functions"""

  def __init__(self, config, head2size):
    super(MultiHeadModel, self).__init__(config, head2size)
    config.num_labels = 1
    self.bert = BertModel(config)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)
    module_dict = {}
    for head_name, num_labels in head2size.items():
      module_dict[head_name] = nn.Linear(config.hidden_size, num_labels)
    self.heads = nn.ModuleDict(module_dict)

    self.init_weights()

  def forward(self, input_ids, token_type_ids=None, attention_mask=None,
              head2labels=None, return_pooler_output=False, head2mask=None,
              nsp_loss_weights=None):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get logits
    output = self.bert(
      input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
      output_attentions=False, output_hidden_states=False, return_dict=True)
    pooled_output = self.dropout(output["pooler_output"]).to(device)

    head2logits = {}
    return_dict = {}
    for head_name, head in self.heads.items():
      head2logits[head_name] = self.heads[head_name](pooled_output)
      head2logits[head_name] = head2logits[head_name].float()
      return_dict[head_name + "_logits"] = head2logits[head_name]


    if head2labels is not None:
      for head_name, labels in head2labels.items():
        num_classes = head2logits[head_name].shape[1]

        # Regression (e.g. for politeness)
        if num_classes == 1:

          # Only consider positive examples
          if head2mask is not None and head_name in head2mask:
            num_positives = head2labels[head2mask[head_name]].sum()  # use certain labels as mask
            if num_positives == 0:
              return_dict[head_name + "_loss"] = torch.tensor([0]).to(device)
            else:
              loss_fct = MSELoss(reduction='none')
              loss = loss_fct(head2logits[head_name].view(-1), labels.float().view(-1))
              return_dict[head_name + "_loss"] = loss.dot(head2labels[head2mask[head_name]].float().view(-1)) / num_positives
          else:
            loss_fct = MSELoss()
            return_dict[head_name + "_loss"] = loss_fct(head2logits[head_name].view(-1), labels.float().view(-1))
        else:
          loss_fct = CrossEntropyLoss(weight=nsp_loss_weights.float())
          return_dict[head_name + "_loss"] = loss_fct(head2logits[head_name], labels.view(-1))


    if return_pooler_output:
      return_dict["pooler_output"] = output["pooler_output"]

    return return_dict

class InputBuilder(object):
  """Base class for building inputs from segments."""

  def __init__(self, tokenizer):
      self.tokenizer = tokenizer
      self.mask = [tokenizer.mask_token_id]

  def build_inputs(self, history, reply, max_length):
      raise NotImplementedError

  def mask_seq(self, sequence, seq_id):
      sequence[seq_id] = self.mask
      return sequence

  @classmethod
  def _combine_sequence(self, history, reply, max_length, flipped=False):
      # Trim all inputs to max_length
      history = [s[:max_length] for s in history]
      reply = reply[:max_length]
      if flipped:
          return [reply] + history
      return history + [reply]


class BertInputBuilder(InputBuilder):
  """Processor for BERT inputs"""

  def __init__(self, tokenizer):
      InputBuilder.__init__(self, tokenizer)
      self.cls = [tokenizer.cls_token_id]
      self.sep = [tokenizer.sep_token_id]
      self.model_inputs = ["input_ids", "token_type_ids", "attention_mask"]
      self.padded_inputs = ["input_ids", "token_type_ids"]
      self.flipped = False


  def build_inputs(self, history, reply, max_length, input_str=True):
    """See base class."""
    if input_str:
        history = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(t)) for t in history]
        reply = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(reply))
    sequence = self._combine_sequence(history, reply, max_length, self.flipped)
    sequence = [s + self.sep for s in sequence]
    sequence[0] = self.cls + sequence[0]

    instance = {}
    instance["input_ids"] = list(chain(*sequence))
    last_speaker = 0
    other_speaker = 1
    seq_length = len(sequence)
    instance["token_type_ids"] = [last_speaker if ((seq_length - i) % 2 == 1) else other_speaker
                                  for i, s in enumerate(sequence) for _ in s]
    return instance


def _initialize(path="."):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    input_builder = BertInputBuilder(tokenizer=tokenizer)
    model = MultiHeadModel.from_pretrained(path, head2size={"nsp": 2})
    model.to(device)
    model.eval()
    return input_builder, device, model
