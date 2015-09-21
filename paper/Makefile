
.PHONY: all clean

BASENAMES=articulation_estimation
TARGETS=$(foreach var,$(BASENAMES),out/$(var).pdf)

# http://stackoverflow.com/questions/18136918/how-to-get-current-directory-of-your-makefile
ROOT_DIR:=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

OUT_DIR = out
DEPS_DIR = out/deps
LATEXMK = latexmk -recorder -use-make -deps
PDFLATEX=pdflatex -interaction=nonstopmode

all : $(TARGETS)

# Include the dependency makefile produced by latexmk
# latexmk -deps-out= is used to record the dependencies
$(foreach file,$(TARGETS),$(eval -include $(DEPS_DIR)/$(notdir file)P))


$(OUT_DIR)/.touch:
	mkdir -p $(dir $@) 
	touch $@

$(DEPS_DIR)/.touch:
	mkdir -p $(dir $@)
	touch $@

out/%.pdf : %.tex $(DEPS_DIR)/.touch $(OUT_DIR)/.touch
	$(LATEXMK) -pdf -dvi- -ps- -output-directory=out -pdflatex="$(PDFLATEX)" -deps-out=$(DEPS_DIR)/$(notdir $@)P $<

clean:
	latexmk -C -output-directory=out
