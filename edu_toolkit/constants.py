VALID_FILE_EXTENSIONS = [".json", ".csv", ".xlsx"]
CONVERSATION_PROMPTS_DIR = "prompts/conversation"
UTTERANCE_PROMPTS_DIR = "prompts/utterance"
RESULTS_DIR = "results"

OUTPUT_TALKTIME_WORDS = "talktime_words"
OUTPUT_TALKTIME_TIMESTAMP = "talktime_timestamp"
OUTPUT_MATH_DENSITY = "math_density"
OUTPUT_UPTAKE = "uptake"
OUTPUT_STUDENT_REASONING = "student_reasoning"
OUTPUT_FOCUSING_QUESTIONS = "focusing_questions"

# OPENAI MODEL CONTEXT LENGTH
# Copied from https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo
OPENAI_MODEL_2_CONTEXT_LENGTH = {
    "gpt-4": 8_192,
    "gpt-4-32k": 32_768,
    "gpt-4-0613": 8_192,
    "gpt-4-32k-0613": 32_768,
    "gpt-3.5-turbo": 4_096,
    "gpt-3.5-turbo-16k": 16_385,
    "gpt-3.5-turbo-instruct": 4_096,
    "gpt-3.5-turbo-0613": 4_096,
    "gpt-3.5-turbo-16k-0613": 16_385,
    "gpt-3.5-turbo-0301": 4_096,
}

# DATASET SPECIFIC
AMBER_DATADIR = "data/amber"
AMBER_SPEAKER_COLUMN = "speaker"
AMBER_TEXT_COLUMN = "dialogue"
AMBER_TEACHER = ["Tutor"]
AMBER_STUDENT = ["Student"]
AMBER_START_TIME_COLUMN = "start"
AMBER_END_TIME_COLUMN = "stop"

NCTE_DATADIR = "data/ncte_sessions"
NCTE_SPEAKER_COLUMN = "speaker"
NCTE_TEXT_COLUMN = "text"
NCTE_TEACHER = ["teacher"]
NCTE_STUDENT = ['student', 'multiple students']

TALKMOVES_DATADIR = "data/talkmoves"
TALKMOVES_SPEAKER_COLUMN = "Speaker"
TALKMOVES_TEXT_COLUMN = "Sentence"
TALKMOVES_TEACHER = ["T"]
TALKMOVES_STUDENT = ["S", "SS"]

# UPTAKE
UPTAKE_HF_MODEL_NAME = "ddemszky/uptake-model"
UPTAKE_MIN_NUM_WORDS_SPEAKER_A = 5
HIGH_UPTAKE_THRESHOLD = 0.8
UPTAKE_MAX_INPUT_LENGTH = 120

# STUDENT REASONING
STUDENT_REASONING_HF_MODEL_NAME = "ddemszky/student-reasoning"
STUDENT_REASONING_MIN_NUM_WORDS = 8
STUDENT_REASONING_MAX_INPUT_LENGTH = 128

# FOCUSING QUESTIONS
FOCUSING_HF_MODEL_NAME = "ddemszky/focusing-questions"
FOCUSING_MIN_NUM_WORDS = 0
FOCUSING_MAX_INPUT_LENGTH = 128

# MATH DENSITY
MATH_PREFIXES = [ 
    "sum",
    "arc",
    "mass",
    "digit",
    "graph",
    "liter",
    "gram",
    "add",
    "angle",
    "scale",
    "data",
    "array",
    "ruler",
    "meter",
    "total",
    "unit",
    "prism",
    "median",
    "ratio",
    "area",
]

MATH_WORDS = [
    "absolute value",
    "area",
    "average",
    "base of",
    "box plot",
    "categorical",
    "coefficient",
    "common factor",
    "common multiple",
    "compose",
    "coordinate",
    "cubed",
    "decompose",
    "dependent variable",
    "distribution",
    "dot plot",
    "double number line diagram",
    "equivalent",
    "equivalent expression",
    "ratio",
    "exponent",
    "frequency",
    "greatest common factor",
    "gcd",
    "height of",
    "histogram",
    "independent variable",
    "interquartile range",
    "iqr",
    "least common multiple",
    "long division",
    "mean absolute deviation",
    "median",
    "negative number",
    "opposite vertex",
    "parallelogram",
    "percent",
    "polygon",
    "polyhedron",
    "positive number",
    "prism",
    "pyramid",
    "quadrant",
    "quadrilateral",
    "quartile",
    "rational number",
    "reciprocal",
    "equality",
    "inequality",
    "squared",
    "statistic",
    "surface area",
    "identity property",
    "addend",
    "unit",
    "number sentence",
    "make ten",
    "take from ten",
    "number bond",
    "total",
    "estimate",
    "hashmark",
    "meter",
    "number line",
    "ruler",
    "centimeter",
    "base ten",
    "expanded form",
    "hundred",
    "thousand",
    "place value",
    "number disk",
    "standard form",
    "unit form",
    "word form",
    "tens place",
    "algorithm",
    "equation",
    "simplif",
    "addition",
    "subtract",
    "array",
    "even number",
    "odd number",
    "repeated addition",
    "tessellat",
    "whole number",
    "number path",
    "rectangle",
    "square",
    "bar graph",
    "data",
    "degree",
    "line plot",
    "picture graph",
    "scale",
    "survey",
    "thermometer",
    "estimat",
    "tape diagram",
    "value",
    "analog",
    "angle",
    "parallel",
    "partition",
    "pentagon",
    "right angle",
    "cube",
    "digital",
    "quarter of",
    "tangram",
    "circle",
    "hexagon",
    "half circle",
    "half-circle",
    "quarter circle",
    "quarter-circle",
    "semicircle",
    "semi-circle",
    "rectang",
    "rhombus",
    "trapezoid",
    "triangle",
    "commutative",
    "equal group",
    "distributive",
    "divide",
    "division",
    "multipl",
    "parentheses",
    "quotient",
    "rotate",
    "unknown",
    "add",
    "capacity",
    "continuous",
    "endpoint",
    "gram",
    "interval",
    "kilogram",
    "volume",
    "liter",
    "milliliter",
    "approximate",
    "area model",
    "square unit",
    "unit square",
    "geometr",
    "equivalent fraction",
    "fraction form",
    "fractional unit",
    "unit fraction",
    "unit interval",
    "measur",
    "graph",
    "scaled graph",
    "diagonal",
    "perimeter",
    "regular polygon",
    "tessellate",
    "tetromino",
    "heptagon",
    "octagon",
    "digit",
    "expression",
    "sum",
    "kilometer",
    "mass",
    "mixed unit",
    "length",
    "measure",
    "simplify",
    "associative",
    "composite",
    "divisible",
    "divisor",
    "partial product",
    "prime number",
    "remainder",
    "acute",
    "arc",
    "collinear",
    "equilateral",
    "intersect",
    "isosceles",
    "symmetry",
    "line segment",
    "line",
    "obtuse",
    "perpendicular",
    "protractor",
    "scalene",
    "straight angle",
    "supplementary angle",
    "vertex",
    "common denominator",
    "denominator",
    "fraction",
    "mixed number",
    "numerator",
    "whole",
    "decimal expanded form",
    "decimal",
    "hundredth",
    "tenth",
    "customary system of measurement",
    "customary unit",
    "gallon",
    "metric",
    "metric unit",
    "ounce",
    "pint",
    "quart",
    "convert",
    "distance",
    "millimeter",
    "thousandth",
    "hundredths",
    "conversion factor",
    "decimal fraction",
    "multiplier",
    "equivalence",
    "multiple",
    "product",
    "benchmark fraction",
    "cup",
    "pound",
    "yard",
    "whole unit",
    "decimal divisor",
    "factors",
    "bisect",
    "cubic units",
    "hierarchy",
    "unit cube",
    "attribute",
    "kite",
    "bisector",
    "solid figure",
    "square units",
    "dimension",
    "axis",
    "ordered pair",
    "angle measure",
    "horizontal",
    "vertical",
    "categorical data",
    "lcm",
    "measure of center",
    "meters per second",
    "numerical",
    "solution",
    "unit price",
    "unit rate",
    "variability",
    "variable",
]

# Talk Moves - Coding Manual: https://github.com/SumnerLab/TalkMoves/blob/main/Coding%20Manual.pdf
# Teacher Talk Moves 
# Model: https://huggingface.co/YaHi/teacher_electra_small
# Paper: https://github.com/SumnerLab/TalkMoves
TEACHER_TALK_MOVES_HF_MODEL_NAME = "YaHi/teacher_electra_small"
TEACHER_TALK_MOVES_LABEL_2_NL = {
    0: "No Talk Move Detected",
    1: "Keeping Everyone Together",
    2: "Getting Students to Related to Another Student's Idea",
    3: "Restating",
    # Note, in the original coding manual, revoicing and pressing for accuracy are flipped (5 and 4); 
    # However, upon inspection of their dataset, it appears that revoicing should be 4 and pressing for accuracy should be 5.
    # Therefore, we have used the following mapping:
    4: "Revoicing",
    5: "Pressing for Accuracy",
    6: "Pressing for Reasoning",
}

# Student Talk Moves
# Model: https://huggingface.co/YaHi/student_electra_small
# Paper: https://github.com/SumnerLab/TalkMoves
STUDENT_TALK_MOVES_HF_MODEL_NAME = "YaHi/student_electra_small"
STUDENT_TALK_MOVES_LABEL_2_NL = {
    0: "No Talk Move Detected",
    1: "Relating to Another Student",
    2: "Asking for More Information",
    3: "Making a Claim",
    4: "Providing Evidence or Reasoning"
}