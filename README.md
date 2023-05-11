# Spliformer

Spliformer is a deep-learning tool based on CNN and self-attention mechanism for predicting the influence of genetic variants on RNA splicing in human and visualizing the attention weight score (AWS) of splicing motifs(more details, see [paper](paperlink)). It can take a VCF file containing variants of interest as input and predict the possibility of a variant causing mis-splicing. In addition, it can draw the AWS heatmaps of splicing motifs in the wild type and variant type sequences for exploring any potential splicing motifs.

Spliformer can also be run on our [website](weblink)(updating), where the researchers could easily predict variants of interests and visualize the splicing motifs with their AWS in the heatmap.
## Prerequisites
```
pyfaidx>=0.6.3.1,
pysam>=0.19.1,
numpy>=1.22.4,
pandas>=1.2.3,
seaborn>=0.11.2,
torch>=1.5.0 #CPU version
torch>=1.9.0 #GPU version
```
Our developing environment has been tested on ```Debian 4.19.235-1 (2022-03-17) x86_64```, Python version is ```python 3.9.6```, and Pytorch GPU version is ```pytorch 1.9.0```.
We encourage user to create a new conda environment before using Spliformer, you can first download miniconda through <https://docs.conda.io/en/latest/miniconda.html>, and then you can create a new conda environment named ```Spliformer```  with ```python 3.9``` through the following commands:
```
conda create -n Spliformer python=3.9
conda activate Spliformer
```

## Installation
- Install pytorch
  * Pytorch-CPU/GPU can be installed via pip or conda. More installation details can be found on <https://pytorch.org/>, under ```INSTALL PYTORCH``` section. 

- Install Spliformer through github repository:
```
git clone https://github.com/TJ-zhanglab/Spliformer.git
cd Spliformer
python setup.py install
```

## Usage
Spliformer can be  run under two modes
> **General mode:**
```sh
#Predict the influence of variants on RNA splicing
spliformer -T general -I ./examples/inputhg19.vcf -O ./output.vcf -R /path/genome.fa -A ./reference/hg19anno.txt
```
> **Motif mode:**
```sh
#Visualize the AWS (attention weight score) of splicing motifs in the wild type and variant type sequences.
spliformer -T motif -I ./examples/inputhg19-motif.vcf -R /path/genome.fa -A ./reference/hg19anno.txt

#The results will be saved in the motif_result folder.
```

**Required parameters**

-   -T: Tools (general/motif)for prediction (default: general)
-   -I: Input VCF with variants of interest.
-   -O: Output VCF with prediction of Spliformer in **general mode**
-   -R: Reference fasta file of human genome. Please download it first before making prediction from [GRCh37/hg19](http://hgdownload.cse.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz) or [GRCh38/hg38](http://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz).
-   -A: Annotation file of human genome.  We created hg19/hg38 genome annotation file according to the GENCODE v39 gtf file. The files locate in the [./reference/](https://github.com/TJ-zhanglab/Spliformer/tree/main/reference).

**Optional parameters**

-   -D: The range of distance between the variant and gained/lost splice site shows in the output.vcf in **general mode**. The range of distance can be chosen is from ```0 to 4999``` (default: 50).
-   -M: Mask predicted scores with annotated acceptor/donor gain and unannotated acceptor/donor loss. ```0: not masked; 1: masked``` (default: 0).
-   -N: Number of motifs represents in the AWS heatmap in **motif mode**. The range of numbers can be chosen is from ```0 to 40``` (default: 10).

Details of Spliformer INFO field in the VCF in **general mode**: Ref>Alt|gene|Increased acceptor dis: score|Decreased acceptor dis: score|Increased donor dis: score|Decreased donor dis: score:

|Name                          |Disciption                         |
|-------------------------------|-----------------------------|
|Ref > Alt            |Reference alle > Alternate allele            |
|gene            |Gene name            |
|Increased acceptor dis: score|The distance of an acceptor with maximum increased possibility away from the variant: its increased score|
|Decreased acceptor dis: score|The distance of an acceptor with maximum decreased possibility away from the variant: its decreased score|
|Increased donor dis: score|The distance of a donor with maximum increased possibility away from the variant: its increased score|
|Decreased donor dis: score|The distance of a donor with maximum decreased possibility away from the variant: its decreased score|

## Examples of output file

>  **General mode:**

An example of input file and its prediction file can be found at [examples/inputhg19.vcf](https://github.com/TJ-zhanglab/Spliformer/tree/main/examples) and [examples/output.vcf](https://github.com/TJ-zhanglab/Spliformer/tree/main/examples) respectively.  The prediction result in output.file ```G>A|TTN|2:0.00|38:0.01|2:0.83|-38:0.31```for the variant ```chr2: 179642185 G>A ```can be interpreted as follows:

-   The possibility of the position chr2: 179642187 (2 bp downstream of the variant) is used as an acceptor increased by 0.00.
-   The probability of the position chr2: 179642223 (38 bp downstream of the variant) is used as an acceptor decreased by 0.01.
-   The probability of the position chr2: 179642187 (2 bp downstream of the variant) is used as a donor increased by 0.83.
-   The probability of the position chr2: 179642147 (38 bp upstream of the variant) is used as a donor decreased by 0.31.

>**Motif mode:**

An example of input file and its prediction file can be found at [examples/inputhg19-motif.vcf](https://github.com/TJ-zhanglab/Spliformer/tree/main/examples) and [examples/motif_results/TTN_motif_aws/](https://github.com/TJ-zhanglab/Spliformer/tree/main/examples/motif_results/TTN_motif_aws) respectively.  The outputs under motif mode are two AWS heatmap of splicing motifs in the wild type and variant type sequence according to the variant’s information provided in the inputhg19-motif.vcf :

From the heatmap, we can find that the ```variant (chr2: 179642185 G>A)``` significantly increased the AWS (from 0.27 to 0.66) of regulatory motif ```AGAAUCACUGGGU``` to target splice motif ```GCCUACCCUGUUU``` in variant type sequence compared with the one in wild type sequence, suggesting that regulatory motif ```AGAAUCACUGGGU``` may play a potential role in RNA splicing:
![image](https://github.com/TJ-zhanglab/Spliformer/blob/main/TTN_motif.png)
## Cite us
If you use Spliformer for prediction, please cite [paper](link)
