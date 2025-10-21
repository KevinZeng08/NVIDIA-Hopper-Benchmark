GENCODE_SM50 := -gencode=arch=compute_50,code=\"sm_50,compute_50\"
GENCODE_SM61 := -gencode=arch=compute_61,code=\"sm_61,compute_61\"
GENCODE_SM70 := -gencode=arch=compute_70,code=\"sm_70,compute_70\"
GENCODE_SM89 := -gencode=arch=compute_89,code=\"sm_89,compute_89\"
GENCODE_SM90 := -gencode=arch=compute_90,code=\"sm_90,compute_90\"
PTXAS_FLAG := --ptxas-options=--verbose,--register-usage-level=10,--warn-on-local-memory-usage
PROFILE_FLAG := -lineinfo

CUOPTS = $(GENCODE_SM90) 


CC := nvcc

INCLUDE := 
LIB := 

SRC = tma_bw.cu

EXE = tma_bw

release:
	$(CC) $(CUOPTS) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(LIB) -lcudart ${PTXAS_FLAG} ${PROFILE_FLAG}

clean:
	rm -f *.o; rm -f $(EXE)

run:
	./$(EXE)
