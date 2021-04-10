## Automatic Speech Recognition using DeepSpeech on Amazon SageMaker
---

**Speech Recognition** is the task of translating and converting spoken language into text. Audio data is virtually very difficult for computer systems to search and analyze. Therefore, recorded speech needs to be converted to text before it can be used in various different applications. Automatic Speech Recognition is the task of using deep supervised learning techniques to automatically and accurately convert speech into text. This code is an example of how to use [DeepSpeech](https://github.com/mozilla/DeepSpeech) library to prepare, build, train and host a model using Amazon SageMaker.

**DeepSpeech** is an open source Speech-To-Text engine based on Baidu’s Deep Speech research [paper](https://arxiv.org/pdf/1412.5567.pdf) which implements the DeepSpeech architecture in Tensorflow.

## Data Preparation
In this example, we will use the common voice dataset format to train an Arabic ASR model but the setup applies to any other language in the common voice dataset format. We will need to run a few steps to prepare the data, alphabet file, language model, and the scorer. Let's start with the data format:

### Data format:
CV dataset has three columns as below:

| wav_filename   |      wave_filesize      |  transcript |
|----------|:-------------:|------:|
| common_voice_ar_22759417.wav |  132908 | سيسعدني مساعدتك أي وقت تحب |
| common_voice_ar_23675091.wav |    101420   |   إنك تكبر المشكلة |
| common_voice_ar_23558552.wav | 222764 |    ليست هناك مسافة على هذه الأرض أبعد من يوم أمس |

### Building the Language Model:

A language model is used to predict what words are more likely to follow each other in a sequence. To build the language model, I am using a python script provided with the Deep Speech library called `generate_lm.py` available [HERE](https://github.com/mozilla/DeepSpeech/blob/master/data/lm/generate_lm.py) 
The script takes in a large corpora of text data `input.txt.gz` and does a few transformation steps into it:
- Converts words to lowercase
- Counts word occurences and saves the top-k most common words to a file.
- Use KenLM Binary files to create the language model binary files

To create the Arabic language model, I run the following command on a large corpora of text:
```python
python3 generate_lm.py --input_txt input.txt.gz --output_dir . --top_k 500000 --kenlm_bins native_client/kenlm/build/bin/ --arpa_order 5 --max_arpa_memory "85%" --arpa_prune "0|0|1" --binary_a_bits 255 --binary_q_bits 8 --binary_type trie —discount_fallback
```

The output of this process is two files: `lm.binary` and `vocab-<top_k>.txt` files

### Generate the scorer package:
The scorer package is a language model that is used to direct the beam search that happens during the decoding process to generate the output characters and words. The `generate_scorer_package` binary is available in the `native_client` package. To install the package, run the command below in the terminal:

```bash
sh-4.2$ mkdir native_clients; cd native_clients ; wget -c https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/native_client.amd64.cuda.linux.tar.xz && tar -Jxvf native_client.amd64.cuda.linux.tar.xz 
```
Once the files are downloaded and unpacked, you can generate the scorer package from the files generated at the previous step:

```bash
sh-4.2$ generate_scorer_package --alphabet ./alphabet.txt --lm lm.binary --vocab vocab-500000.txt --package kenlm.scorer --default_alpha 0.6560092006459668 --default_beta 2.3034529727156823
```

The `default_alpha` and `default_beta` values are used to assign initial wieghts to a sequence of words. The can be tuned and optimized after training a model. 


## Start Training
Refer to the [Notebook](notebook/DeepSpeech-SageMaker.ipynb) for training and preparing the SageMaker container.

## Roadmap
- ~~Build local Inference.~~ (Completed)
- Build a SageMaker Inference (WIP)
- Include steps to optimize inference requests (tuning lm_alpha and lm_beta values involved in creating scorer) (WIP)
- Multi-instance GPU training (WIP)
- Support fine tuning.
- Support transfer learning.
- Add steps to create alphabet, vocab files for any language.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This code is licensed under the MIT-0 License. See the LICENSE file.

