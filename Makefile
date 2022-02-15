COMPILER = latexmk
presentacion_flags = -pdf
clean_flags = -c

TARGETS = presentacion

all: $(TARGETS)

presentacion: presentacion.tex
	$(COMPILER) $(presentacion_flags) $<

clean:
	$(COMPILER) $(clean_flags)
	rm -r *.nav *.snm 

.PHONY: clean all
