
  !!! THIS IS *NOT* THE ORIGINAL BZIP2 SOURCE CODE !!!

!!! THIS BZIP2 HAS BEEN MODIFIED FOR EXECUTION ON GPU !!!

This Bzip2 source has been modified to support All-Core Burrows Wheeler Compression as 
described in the paper: 

Aditya Deshpande and P J Narayanan, "[Fast Burrows Wheeler Compression Using All-Cores](http://ardeshp2.web.engr.illinois.edu/papers/Aditya2015Ashes.pdf)", IPDPSW'2014. 

The code is developed on Seward's popular open-source Bzip2 file compressor. The 
partial sorts of Burrows Wheeler Transform step are performed on the GPU and merge,
mtf and huffman steps are performed on the CPU in overlapped manner. The idle
CPU cores use the state-of-the-art BWC by Seward and inter-block parallelism to
process other blocks. The code has been written by 

Aditya Deshpande at Center for Visual Information Technology, International Institute of 
Information Technology, Gachibowli, Hyderabad.

Datasets are available on CVIT Resources (http://cvit.iiit.ac.in/research/resources) page 
and can be downloaded from publicly available URLs also

1. Silesia Data Corpus ( URL: http://sun.aei.polsl.pl/sdeor/index.php?page=silesia )
2. Enwiki-latest-abstract10.xml ( URL: http://dumps.wikimedia.org/enwiki/latest/ )
3. linux-2.6.11.tar ( URL: http://www.kernel.org/pub/linux/kernel/v2.6/ )
4. enwik8 (URL: http://cs.fit.edu/~mmahoney/compression/textdata.html)

The details of this All-Core BWC software are as follows. 

Do a make to create the executable 'bzip2' in the current directory. 

```
make
```

To remove the installed bzip2 from the current directory. 

```
make clean
```

To run on test file, which in this case is README_bzip2.

```
bash run.sh -c 9 0 README_bzip2
```
 
To use this code, it would be better to use the run.sh script. 

```
bash run.sh -c <blocksize100K> <additional cpu threads> <filename>

-c stands for check by doing decompression of compressed file and diff with the original file

<blocksize100K> : blocksize in multiples of 100K, 9 will stand for 900K, 45 for ~4.5MB, 90 for ~9MB etc.

<additional cpu threads>: set to 0 if only CPU+GPU thread is to be run, 
			  set to 'k' if k additional CPU threads are to be run with 1 CPU+GPU thread
```


Important Notice

```
THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS
OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

If you use this code, please cite

```
@INPROCEEDINGS{DeshpandeFastBurrowsWheeler15, 
  author={Aditya Deshpande and P. J. Narayanan}, 
  booktitle={2015 IEEE International Parallel and Distributed Processing Symposium Workshop}, 
  title={Fast Burrows Wheeler Compression Using All-Cores}, 
  year={2015}, 
  pages={628-636}, 
  month={May}}
```
