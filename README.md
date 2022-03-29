#   Translate text from VTT file into speech using Microsoft Cognito API

###  About

This package allows you to translate text in VTT file to speech and output WAV files for each text segment.
It can also output a combined WAV file with all the WAV segments alligned to correct time positions.
You can also automatically fix segements overlap so that the resulting file can be directly imported to your video.

If you want to use Adobe Audition you can output an XML file that you can import directly into the program.

---

Author: Cyprian Vero

Date: 29 March 2022


###  Installation

Tested on Python 3.8

If you have conda
```sh
conda create -n py38 python=3.8 -y
conda activate py38
```
Required packages:
```sh
pip install natsort matplotlib ffmpeg azure-cognitiveservices-speech pydub tqdm
```

###  Usage

To run the tranlation use the following file:

```sh
python translate_vtt.py
```

**Input:** 
* .vtt file

**Outputs:**
* wav file for each of the text segments in a VTT file
* adobe_audition_output_original.xml

(_optional_) when flag `--auto_remove_overlap` is used:
* combined wav file of all segments adjusted to correct time placement and corrected for any overlaps.
* adjusted wav files
* adobe_audition_output_adjusted.xml 

### Example 

Translate a VTT file called french.vtt to speach with an automatic correction of overlapping files
```sh
python translate_vtt.py --file french.vtt --language "fr-FR" --voice "fr-FR-HenriNeural" --API_key "[TYPE_YOUR_API_KEY_HERE]" --API_region "westeurope" --auto_remove_overlap
```

### Options

**Required**

```sh
--language (type=str)
  Speech language to translate text to. Ex. "fr-FR" for French. Full list available at: https://docs.microsoft.com/en-us/azure/cognitive-services/speech-service/language-support#prebuilt-neural-voices

--voice (type=str)
  The voice to be used for speech. Ex. "fr-FR-HenriNeural" for French. Full list available at: https://docs.microsoft.com/en-us/azure/cognitive-services/speech-service/language-support#prebuilt-neural-voices

--API_key (type=str)
  A translation API_Key from Microsoft Cognito website.

--API_region (type=str)
  A translation API_region from Microsoft Cognito website. Ex. "westeurope" for Western Europe
```

To find available languages go to https://docs.microsoft.com/en-us/azure/cognitive-services/speech-service/language-support#prebuilt-neural-voices


**Optional**

```sh
--file (type=str) default='./test.vtt',
  File path to vtt file that should be translated.

--output_folder (type=str) default='./audio_files/',
  Directory path to the outputs folder.

--allowed_overlap_milliseconds (type=int) default=50,
  Maximum number of milliseconds one translation track can overlap the next translation track

--auto_remove_overlap (action='store_true')
  Automatically speed up the the segment to fit the available space without overlap. If a track 1 overlaps track 2 by 1000 ms then the track 1 length will be speedup by 1000 ms.
    
--use_existing_translations (action='store_true')
  Used for debugging. Instead of translating via API, combine files that are already translated and available in the --output_folder.
```



