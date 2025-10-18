GENCODE_SM50 := -gencode=arch=compute_50,code=\"sm_50,compute_50\"
GENCODE_SM61 := -gencode=arch=compute_61,code=\"sm_61,compute_61\"
GENCODE_SM70 := -gencode=arch=compute_70,code=\"sm_70,compute_70\"
GENCODE_SM89 := -gencode=arch=compute_89,code=\"sm_89,compute_89\"
GENCODE_SM90 := -gencode=arch=compute_90,code=\"sm_90,compute_90\"
PTXAS_FLAG := --ptxas-options=--verbose,--register-usage-level=10,--warn-on-local-memory-usage

CUOPTS = $(GENCODE_SM90) 


CC := nvcc

INCLUDE := 
LIB := 

SOURCES = cp_async_2d.cu
EXES = cp_async_2d

all: $(EXES)

cp_async_2d: cp_async_2d.cu
	$(CC) $(CUOPTS) $< -o $@ -I$(INCLUDE) -L$(LIB) -lcudart ${PTXAS_FLAG}

release: all

clean:
	rm -f *.o $(EXES)

run: all
	./cp_async_2d
