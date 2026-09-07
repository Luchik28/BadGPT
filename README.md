## Welcome to badGPT

A (poorly) made LLM. It is made with just Python and no external libraries/dependencies (except for numPy). You can check out a demo [here](https://luchik28.github.io/BadGPT/).

I need numPy for efficient matrix multiplication, as trying to do that in just python would be too slow (each training pass would take hours as opposed to like seconds). numPy does this in C, which is faster, but trying to write that out myself is a bit out of the scope for this project so I'm just using numPy.

### How to use

Download the files, use pip to install numPy, and run each file. It's not organized very well (and named quite poorly), but in general I made a new file for each new video in his series, and didn't remove old files.

If you don't want to download anything, go check out the [demo](https://luchik28.github.io/BadGPT/). It has 4 tabs that each showcase one of the projects I did, even though they all build on eachother. Each tab includes a bit of info on the project, and how it is built.

### Project structure.

The docs folder includes files for the demo website. Index.html is the code for the actual website, the json files include the stored weights for each file, and the javascript files reconstruct the model using the weights for the demo to use.

The plots folder contains an exported image of how the model embedded each letter into 2 dimensions, and how that kind of shows how it embeds meaning (t and h are relatively close, for example).

The Jupyter Notebooks are where I did most of the work. The first (makemoreWithJustMath.py, makemore3.ipynb, makemore5wavenet.ipynb) include all dependencies (except numPy and some other things) in the actual notebook, they're mostly me following Andrej Karpathy's tutorials. After that (GPT.ipynb, GPTwithTokenizer.ipynb, and GPTwithTokenizernoplot.ipynb), I put many of the classes (such as Value, Layer, etc) into another python file (wavenetArchitechture.py), that is included as a dependency on the notebooks. While it started out as just classes needed for the wavenet architecture, I added a bunch of other things for later work, and didn't update the name.

names.txt, shakespear.txt, and fineweb_edu_subset.txt are all datasets to train on. exportFineWebWeights.py and exportGPTweights.py include functions to export the parameters from the notebooks to the json files that the demo website uses to show the models. fetch_fineweb_subset.py works to construct the fineweb data from huggingface. bpeTokenizer.py includes the byte pair encoding tokenizer code, and functions to easily use it.

All other files probably don't do much.

### Credits

It is heavily inspired by Andrej Karpathy's Micrograd/makeMore series.
I got the names dataset and the shakespear dataset from various Github repositories related to Karpathy's series, and the Fineweb dataset from [hugging face](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu).
