###############################################################################################

#   Translate text from VTT file into speech

#   Input: .vtt file
#   Outputs:
#       - wav file for each of the segments in VTT file
#       - combined wav file of all segments with auto adjusted placement for overlaps.
#       - adobe_audition_output_original.xml
#       - (optional) adjusted wav files if --auto_remove_overlap was used
#       - (optional) adobe_audition_output_adjusted.xml if --auto_remove_overlap was used

#   By: Cyprian Vero

#   Date: March 28th 2022

#   EXAMPLE USAGE:

#   python translate_vtt.py --language "fr-FR" --voice "fr-FR-HenriNeural" --API_key "[TYPE_YOUR_API_KEY_HERE]" --API_region "westeurope" --auto_remove_overlap --use_existing_translations

###############################################################################################

import argparse
import os
import pprint
from tqdm import tqdm
import re
from pydub import AudioSegment
import math

import azure.cognitiveservices.speech as speechsdk

def parse_args():
    desc = "Translate the text from VTT file to speech using Microsoft Cognito service and automatically join the translations into a single WAV file." 
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--language', type=str,
        help='Speech language to translate text to. Ex. "fr-FR" for French. Full list available at: https://docs.microsoft.com/en-us/azure/cognitive-services/speech-service/language-support#prebuilt-neural-voices')

    parser.add_argument('--voice', type=str,
        help='The voice to be used for speech. Ex. "fr-FR-HenriNeural" for French. Full list available at: https://docs.microsoft.com/en-us/azure/cognitive-services/speech-service/language-support#prebuilt-neural-voices')

    parser.add_argument('--API_key', type=str,
        help='Provide your translation API_Key from Microsoft Cognito website.')
    
    parser.add_argument('--API_region', type=str,
        help='Provide your translation API_region from Microsoft Cognito website. Ex. "westeurope" for Western Europe')
 
    parser.add_argument('--file', type=str,
        default='./test.vtt',
        help='File path to vtt file that should be translated. (default: %(default)s)')

    parser.add_argument('--output_folder', type=str,
        default='./audio_files/',
        help='Directory path to the outputs folder. (default: %(default)s)')

    parser.add_argument('--allowed_overlap_milliseconds', type=int,
        default=50,
        help='Maximum number of milliseconds one translation track can overlap the next translation track(default: %(default)s)')
    
    parser.add_argument('--auto_remove_overlap',
        action='store_true',
        help='Automatically speed up the the segment to fit the available space without overlap. If a track 1 overlaps track 2 by 1000 ms then the track 1 length will be speedup by 1000 ms. %(default)s)')
    
    parser.add_argument('--use_existing_translations',
        action='store_true',
        help='Instead of translating via API, combine files that are already translated and available in the --output_folder. %(default)s)')

    
    args = parser.parse_args()
    return args


######################
# HELPER METHODS
######################

import matplotlib.pyplot as plt
import numpy as np
import wave

def show_wave_visualization(file):
    raw = wave.open(file)
     
    signal = raw.readframes(-1)
    signal = np.frombuffer(signal, dtype ="int16")
     
    f_rate = raw.getframerate()
 
    time = np.linspace(
        0, # start
        len(signal) / f_rate,
        num = len(signal)
    )
 
    plt.figure(1)
    plt.title("Sound Wave")
    plt.xlabel("Time")
    
    plt.plot(time, signal)
    
    file_name = os.path.basename(file)
    file_name = str(file)[:-4]+'_sound_visualization'

    plt.savefig(file_name)

######################
# VTT PARSING METHODS
######################

def parse_vtt_file(file):
    opened_file = open(file,encoding='utf8')
    content = opened_file.read()
    parts = content.split('\n\n') # split on double line

    # wrangle segments
    m = re.compile(r"\<.*?\>") # strip/remove unwanted tags

    new_parts = [clean(s,m) for s in parts if len(s)!=0][1:] #skip first line

    start_times = []
    end_times = []
    texts = []
    for part in tqdm(new_parts):
        split_part = part.split('\n')

        time_code = split_part[0]
        split_time_code = time_code.split()
        start_times.append(time_in_miliseconds(split_time_code[0]))
        end_times.append(time_in_miliseconds(split_time_code[1]))

        text = split_part[1]
        texts.append(text)

    return(texts, start_times, end_times)
    
def clean(content, m):
    new_content = m.sub('',content)
    new_content = new_content.replace('-->','')
    return new_content


def time_in_miliseconds(time):
    time = time.split(':')
    hours = time[0] #discard
    minutes = int(time[1])*60*1000
    seconds = int(time[2].split('.')[0]) * 1000
    miliseconds = int(time[2].split('.')[1])

    return minutes+seconds+miliseconds

######################
# TRANSLATION METHODS
######################

def translate_and_save_text(texts, destination):

    #create folder for generated translation files
    root_output_folder = destination
    if not os.path.exists(root_output_folder):
        os.makedirs(root_output_folder)

    output_folder = os.path.join(root_output_folder, 'original')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    speech_config = speechsdk.SpeechConfig(subscription=args.API_key, region=args.API_region)
    # Note: if only language is set, the default voice of that language is chosen.
    speech_config.speech_synthesis_language = args.language # For example, "de-DE"
    # The voice setting will overwrite the language setting.
    # The voice setting will not overwrite the voice element in input SSML.
    # Full list of voices here: https://docs.microsoft.com/en-us/azure/cognitive-services/speech-service/language-support#text-to-speech
    speech_config.speech_synthesis_voice_name =args.voice

    translations = []
    i = 0
    for text in tqdm(texts):

        filename = str(i) + '.wav'
        tranlation_file_path = os.path.join(output_folder, filename)
        audio_config = speechsdk.audio.AudioOutputConfig(filename=tranlation_file_path)
        translations.append(tranlation_file_path)

        #translate text with Microsoft
        translate_text_to_speach(text, speech_config, audio_config)
        i += 1
    
    return translations

def translate_text_to_speach(text, speech_config, audio_config):
    synthesizer = speechsdk.SpeechSynthesizer(speech_config, audio_config)
    synthesizer.speak_text_async(text)

from natsort import natsorted
def load_translations_from_folder(folder):

    accepted_file_types = [".wav"]
    translations = []

    for filename in tqdm(natsorted(os.listdir(folder))):

        file_path = os.path.join(folder, filename)
        if filename.endswith(tuple(accepted_file_types)):
            translations.append(file_path)

    return translations


######################
# AUDIO METHODS
######################

#note: the outcome will be about 25-50ms longer then exptedcted b/c of the way that it calcultes shrinkage
def speedup_segment_by_miliseconds(segment, ms):
    current_width = len(segment)
    new_width = current_width - ms
    width_change_ratio = current_width/new_width

    return (segment.speedup(playback_speed=width_change_ratio, chunk_size=50, crossfade=25), width_change_ratio)

def trim_end_of_segment(segment, trim_in_miliseconds):
    return segment[0:len(segment)-trim_in_miliseconds]

def load_audio_segments_from_files(wave_files, trim_end_ms):
    segments = []
    for wav_path in wave_files:
        #add the space to the output
        segment = AudioSegment.from_file(wav_path, format="wav")
        segment = trim_end_of_segment(segment, trim_end_ms) 
        segments.append(segment)
    return segments

def check_for_overlaps(segments, start_times, auto_shrink=False, allowed_overlap=1):
    adjusted_segments = []

    has_overlap = False
    overlap_info = '\n'
    
    overlaps = 0
    position = 0
    for segment in tqdm(segments):
        adjusted_segments.append(segment)
        end_position_of_current_segment = start_times[position] + len(segment)
        
        if position < len(segments)-1 and (end_position_of_current_segment - allowed_overlap) > start_times[position+1]:
            has_overlap = True
            overlaps += 1
            start_position_of_next_segment = start_times[position+1]
            overlap =  end_position_of_current_segment - start_position_of_next_segment
            overlap_seconds = ((end_position_of_current_segment - start_position_of_next_segment)/1000.0)%60

            if auto_shrink:
                adjusted_segment, width_change_ratio = speedup_segment_by_miliseconds(segment, overlap)
                adjusted_segments[position] = adjusted_segment
                overlap_info = overlap_info + "\n[FIXED] File number " + str(position+1) + " was overlapping file number " + str(position+2) + " by " + str(overlap_seconds) + " seconds.\n\tFile was auto sped up by " + str(int((width_change_ratio - 1)*100))+ "%, and is now "+str(overlap_seconds)+" shorter. There is no overlap anymore.\n\n"
            else:
                overlap_info = overlap_info + "\n[OVERLAP] File number " + str(position+1) + " overlaps the file number " + str(position+2) + " by " + str(overlap_seconds) + " seconds.\n\n"

        position += 1
        
    if has_overlap:
        print(overlap_info)

    #return adjusted segments
    return (adjusted_segments, overlaps)

def save_adjusted_translations(segments, destination):

    #create folder for generated translation files
    root_output_folder = destination
    if not os.path.exists(root_output_folder):
        os.makedirs(root_output_folder)

    output_folder = os.path.join(root_output_folder, 'adjusted')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    paths = []
    i = 0
    for segment in segments:
        filename = str(i) + '.wav'
        file_path = os.path.join(output_folder, filename)
        paths.append(file_path)
        segment.export(file_path, format="wav")
        i += 1
    
    return paths

def translate_text_to_speach(text, speech_config, audio_config):
    synthesizer = speechsdk.SpeechSynthesizer(speech_config, audio_config)
    synthesizer.speak_text_async(text)


#Combine segments at positions from VTT file
def combine_segments(segments, start_times):

    max_length = len(segments)
    position = 0
    combined_segments = AudioSegment.empty()
    for segment in tqdm(segments):
        #calculate silent space between previous and current file
        if position == 0:
            silence_duration = start_times[position]
        elif position < max_length:
            silence_duration = start_times[position]-len(combined_segments)
            
        position += 1

        #add the space to the output
        if silence_duration > 0:
            silence = AudioSegment.silent(duration=silence_duration)
            combined_segments = combined_segments + silence
        
        combined_segments = combined_segments + segment

    return combined_segments




######################
# XML METHODS
######################

import xml.etree.ElementTree as ET
from xml.dom import minidom

def generate_Adobe_Audition_FCP_XML(segments, audio_folder, start_times, file_name):

    tree = ET.Element('xmeml', {'version': '5'})

    project = ET.SubElement(tree, 'project')
    ET.SubElement(project, 'name').text = file_name

    children = ET.SubElement(project, 'children')
    sequence = ET.SubElement(children, 'sequence', {'id': 'sequence-1'})
    ET.SubElement(sequence, 'duration').text = '900'

    rate = ET.SubElement(sequence, 'rate')
    ET.SubElement(rate, 'timebase').text = '30'
    ET.SubElement(rate, 'ntsc').text = 'FALSE'

    ET.SubElement(sequence, 'name').text = file_name

    media = ET.SubElement(sequence, 'media')
    audio = ET.SubElement(media, 'audio')

    _format = ET.SubElement(audio, 'format')
    add_samplecharacteristics_xml_element(_format, '32', '48000')

    add_outputs_xml_element(audio)

    # files_directory = os.path.join(audio_folder, 'adjusted')
    file_names = []
    for name_index in range(len(start_times)):
         file_names.append(str(name_index) + '.wav')
    add_track_xml_element(audio, audio_folder, file_names, start_times, segments, index=1)

    timecode = ET.SubElement(sequence, 'timecode')
    t_rate = ET.SubElement(timecode, 'rate')
    ET.SubElement(t_rate, 'timebase').text = '30'
    ET.SubElement(t_rate, 'ntsc').text = 'FALSE'
    ET.SubElement(timecode, 'frame').text = '0'
    ET.SubElement(timecode, 'displayformat').text = 'NDF'

    i = 0
    for start_time in start_times:
        add_marker_xml_element(sequence=sequence, name='Marker for file '+str(i), _in=str(convert_time_to_frame(start_time, fps=30)), comment='This is comment')
        i += 1


    xmlstr = minidom.parseString(ET.tostring(tree)).toprettyxml(indent="  ")
    xmlstr = xmlstr.replace('<?xml version="1.0" ?>','')

    with open(file_name, 'wb') as f:
        f.write('<?xml version="1.0" encoding="UTF-8" standalone="no" ?>\n<!DOCTYPE xmeml>\n'.encode('utf8'))
        f.write(xmlstr.encode('utf-8'))

def add_samplecharacteristics_xml_element(parent, depth, samplerate):
    samplecharacteristics = ET.SubElement(parent, 'samplecharacteristics')
    ET.SubElement(samplecharacteristics, 'depth').text = depth
    ET.SubElement(samplecharacteristics, 'samplerate').text = samplerate

def add_outputs_xml_element(audio):
    outputs = ET.SubElement(audio, 'outputs')
    add_groups_xml_element(outputs, index='1', channels='1', downmix='0')
    add_groups_xml_element(outputs, index='2', channels='1', downmix='0')
    
def add_groups_xml_element(outputs, index, channels, downmix):
    group = ET.SubElement(outputs, 'group')
    ET.SubElement(group, 'index').text = index
    ET.SubElement(group, 'numchannels').text = channels
    ET.SubElement(group, 'downmix').text = downmix
    channel = ET.SubElement(group, 'channel')
    ET.SubElement(channel, 'index').text = index

def add_track_xml_element(audio, audio_folder, file_names, start_times, segments, index):
    track = ET.SubElement(audio, 'track')
    ET.SubElement(track, 'enabled').text = 'TRUE'
    ET.SubElement(track, 'locked').text = 'FALSE'

    i = 0
    for file_name in file_names:
        start = convert_time_to_frame(start_times[i], fps=30)
        duration = convert_time_to_frame(len(segments[i]), fps=30)
        end = start + duration
        file_path = os.path.join(audio_folder, file_name)
        add_clipitem_xml_element(track, file_path=file_path, id=str(i), name=file_name, duration=str(duration), start=str(start), end=str(end), track_index=str(index))
        i += 1

    ET.SubElement(track, 'outputchannelindex').text = '25'

def add_clipitem_xml_element(track, file_path, id, name, duration, start, end, track_index):
    clip_id = 'clipitem-'+ id
    clipitem = ET.SubElement(track, 'clipitem', {'id': clip_id})
    ET.SubElement(clipitem, 'name').text = name
    ET.SubElement(clipitem, 'enabled').text = 'TRUE'
    ET.SubElement(clipitem, 'duration').text = duration
    ET.SubElement(clipitem, 'start').text = start
    ET.SubElement(clipitem, 'end').text = end
    ET.SubElement(clipitem, 'in').text = '0'
    ET.SubElement(clipitem, 'out').text = duration

    add_translation_file_xml_element(clipitem, id=id, file_name=name , file_path=file_path, duration=duration)

    sourcetrack = ET.SubElement(clipitem, 'sourcetrack')
    ET.SubElement(sourcetrack, 'mediatype').text = 'audio'
    ET.SubElement(sourcetrack, 'trackindex').text = track_index
    ET.SubElement(clipitem, 'channelcount').text = '1'

def add_translation_file_xml_element(clipitem, id, file_name, file_path, duration):
    file = ET.SubElement(clipitem, 'file', {'id': id})
    ET.SubElement(file, 'name').text = file_name
    ET.SubElement(file, 'pathurl').text = file_path
    rate = ET.SubElement(file, 'rate')
    ET.SubElement(rate, 'timebase').text = '30'
    ET.SubElement(rate, 'ntsc').text = 'FALSE'
    ET.SubElement(file, 'duration').text = duration
    media = ET.SubElement(file, 'media')
    audio = ET.SubElement(media, 'audio')
    add_samplecharacteristics_xml_element(audio, '32', '48000')


def add_marker_xml_element(sequence, name, _in, comment):
    marker = ET.SubElement(sequence, 'marker')
    ET.SubElement(marker, 'name').text = name
    ET.SubElement(marker, 'in').text = _in
    ET.SubElement(marker, 'out').text = '-1' # -1 means it does not end
    ET.SubElement(marker, 'comment').text = comment

#convert start position milliseconds to frame number for the marker
def convert_time_to_frame(ms, fps):
    seconds = ms/1000
    return math.floor(seconds * fps)

######################
# WORKFLOW METHODS
######################
                   
def tranlate_vtt_file(file):

    print("\nReading VTT file:\n")
    texts, start_times, end_times = parse_vtt_file(file)

    if args.use_existing_translations:
        print("\nLoading translations from folder: " + args.output_folder + "\n")
        translations = load_translations_from_folder(os.path.join(args.output_folder, 'original'))
    else:
        print("\nTranlating texts:\n")
        translations = translate_and_save_text(texts, destination=args.output_folder)

    #Convert file paths to AudioSegments
    segments_original = load_audio_segments_from_files(translations, trim_end_ms=800) #Microsoft translation files have unwanted 800ms of silence at the end

    #Detect files overlap
    print("\nChecking if new translations will not overlap each other:\n")
    segments_adjusted, overlaps = check_for_overlaps(segments_original, start_times, auto_shrink=args.auto_remove_overlap, allowed_overlap=args.allowed_overlap_milliseconds)

    if overlaps > 0 and not args.auto_remove_overlap:
        erro_msg = "\n[ERROR] There are " + str(overlaps) + " overlap(s) in your files. Fix them manually by rewriting translation text or use --auto_remove_overlap flag to automatically remove overlays by speeding the translation file.\n"
        print(erro_msg)
    else:
        #combine files into one wave
        print("\nCombining audio files into a single file:\n")

        combined_segments = combine_segments(segments_adjusted, start_times)
        combined_segments_file_path = str(file)[:-3]+'wav'
        combined_segments.export(combined_segments_file_path, format="wav")

        save_adjusted_translations(segments_adjusted, args.output_folder)
        show_wave_visualization(combined_segments_file_path)
        generate_Adobe_Audition_FCP_XML(segments_adjusted, audio_folder=os.path.join(args.output_folder, 'adjusted'),  start_times=start_times, file_name="adobe_audition_output_adjusted.xml")

    generate_Adobe_Audition_FCP_XML(segments_original, audio_folder=os.path.join(args.output_folder, 'original'),  start_times=start_times, file_name="adobe_audition_original.xml")


def main():
    global args

    args = parse_args()
    accepted_file_types = [".vtt"]
    
    print("\n-----------START------------\n")
    print("CONFIG:\n")
    pprint.pprint(vars(args))
    print("\n")

    if not args.language:
        print('[ERROR] Missing argument --language. Please provide the Speech language to translate text to. Ex. "fr-FR" for French. Full list available at: https://docs.microsoft.com/en-us/azure/cognitive-services/speech-service/language-support#prebuilt-neural-voices')
    elif not args.voice:
        print('[ERROR] Missing argument --voice. Please provide the translation voice name to be used for speech. Ex. "fr-FR-HenriNeural" for French. Full list available at: https://docs.microsoft.com/en-us/azure/cognitive-services/speech-service/language-support#prebuilt-neural-voices')
    elif not args.API_key:
        print('[ERROR] Missing argument --API_key. Provide your translation API_Key from Microsoft Cognito website.')
    elif not args.API_region:
        print('[ERROR] Missing argument --API_region. Provide your translation API_region from Microsoft Cognito website. Ex. "westeurope" for Western Europe')
    elif args.file.endswith(tuple(accepted_file_types)):
        tranlate_vtt_file(args.file)
    else:
        print('ERORR: This program currently only supports VTT file format\n')
    
    print("\n------------END-----------\n")


if __name__ == "__main__":
	main()