# опции. Их можно изменять при запуске: make run N=4 X=100 Z=100 S=500 D=0.2
N = 32     # количество процессов
L = 1      # индекс геометрии
X = 16     # число вычислительный узлов по X1
Y = 16     # число вычислительный узлов по X2
Z = 32     # число вычислительный узлов по X3
P = 50     # шаг печати
S = 100   # количество шагов
D = 1.0    # delta
K = 1.0    # kappa
C = 0.3    # cfl

# инициализация опций
NUM_PROC       = $(N)
INDEX_GEOMETRY = $(L)
NUM_NODE_X     = $(X)
NUM_NODE_Y     = $(Y)
NUM_NODE_Z     = $(Z)
INTER_PRINT    = $(P)
NUMBER_STEPS   = $(S)
DELTA          = $(D)
KAPPA          = $(K)
CFL            = $(C)

INPUT_DIR   = input
INPUT_FILE  = 
OUTPUT_DIR  = output
OUTPUT_FILE = out

INPUT  = $(INPUT_DIR)/$(INPUT_FILE)
OUTPUT = $(OUTPUT_DIR)/$(OUTPUT_FILE)

ARGUMENTS   = --index-geometry $(INDEX_GEOMETRY) \
				-x $(NUM_NODE_X) -y $(NUM_NODE_Y) -z $(NUM_NODE_Z) \
				--interval-print $(INTER_PRINT) --number-steps $(NUMBER_STEPS) \
				--delta $(DELTA) --kappa $(KAPPA) --cfl $(CFL) \
				--input $(INPUT) --output $(OUTPUT)

# параметры запуска
QUEUE       = test
TIME        = 10:00

# версия DEBUG или RELEASE
VERSION        = c89
VERSION_NUMBER = 1.0.0

# имя бинарника
PROG_NAME = kovcheg
