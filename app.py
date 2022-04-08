'''
###############################################################################################

Translate text from VTT file into speech

Input: .vtt file
Outputs:
    - wav file for each of the segments in VTT file
    - combined wav file of all segments with auto adjusted placement for overlaps.
    - adobe_audition_output_original.xml
    - (optional) adjusted wav files if "Automatically remove overlaps" is checked
    - (optional) adobe_audition_output_adjusted.xml if "Automatically remove overlaps" is checked

By: Cyprian Vero

Date: March 28th 2022

EXAMPLE USAGE:

First run in terminal: streamlit run app.py
Then go to this link http://localhost:8501/

xcode-select --install
pip install watchdog

###############################################################################################
'''

import streamlit as st
import pandas as pd
import argparse
import os
import re
from pydub import AudioSegment
import math
from io import StringIO

import azure.cognitiveservices.speech as speechsdk

def parse_args():
    desc = "Translate the text from VTT file to speech using Microsoft Cognito service and automatically join the translations into a single WAV file." 
    parser = argparse.ArgumentParser(description=desc)

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
    parts = file.split('\n\n') # split on double line

    # wrangle segments
    m = re.compile(r"\<.*?\>") # strip/remove unwanted tags

    new_parts = [clean(s,m) for s in parts if len(s)!=0][1:] #skip first line

    start_times = []
    end_times = []
    texts = []
    for part in new_parts:
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

def translate_and_save_text(texts, destination, api_key, api_region, language, voice):

    progress_bar = st.progress(0)

    #create folder for generated translation files
    root_output_folder = destination
    if not os.path.exists(root_output_folder):
        os.makedirs(root_output_folder)

    output_folder = os.path.join(root_output_folder, 'original')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    speech_config = speechsdk.SpeechConfig(subscription=api_key, region=api_region)
    # Note: if only language is set, the default voice of that language is chosen.
    speech_config.speech_synthesis_language = language # For example, "de-DE"
    speech_config.speech_synthesis_voice_name = voice

    translations = []
    i = 0
    for text in texts:

        filename = str(i) + '.wav'
        tranlation_file_path = os.path.join(output_folder, filename)
        audio_config = speechsdk.audio.AudioOutputConfig(filename=tranlation_file_path)
        translations.append(tranlation_file_path)

        translate_text_to_speach(text, speech_config, audio_config)

        i += 1
        progress_bar.progress(i/len(texts))
    
    return translations

def translate_text_to_speach(text, speech_config, audio_config):
    synthesizer = speechsdk.SpeechSynthesizer(speech_config, audio_config)
    synthesizer.speak_text_async(text)

from natsort import natsorted
def load_translations_from_folder(folder):

    accepted_file_types = [".wav"]
    translations = []

    # progress_bar = st.progress(0)
    with st.spinner('Loading translation audio files...'):
        for filename in natsorted(os.listdir(folder)):

            file_path = os.path.join(folder, filename)
            if filename.endswith(tuple(accepted_file_types)):
                translations.append(file_path)

    st.success('Done!')

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
    progress_bar = st.progress(0)
    for segment in segments:
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
                overlap_info = overlap_info + "**[FIXED]** File number " + str(position+1) + " was overlapping file number " + str(position+2) + " by **" + str(overlap_seconds) + " seconds**.\n\n\tFile was auto sped up by " + str(int((width_change_ratio - 1)*100))+ "%, and is now "+str(overlap_seconds)+" shorter. There is no overlap anymore.\n\n"
            else:
                overlap_info = overlap_info + "**[OVERLAP]** File number " + str(position+1) + " overlaps the file number " + str(position+2) + " by **" + str(overlap_seconds) + " seconds**.\n\n"

        position += 1
        progress_bar.progress(position/len(segments))
        
    if has_overlap:
        st.warning(overlap_info)

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
    progress_bar = st.progress(0)
    max_length = len(segments)
    position = 0
    combined_segments = AudioSegment.empty()
    for segment in segments:
        #calculate silent space between previous and current file
        if position == 0:
            silence_duration = start_times[position]
        elif position < max_length:
            silence_duration = start_times[position]-len(combined_segments)
            
        #add the space to the output
        if silence_duration > 0:
            silence = AudioSegment.silent(duration=silence_duration)
            combined_segments = combined_segments + silence
        
        combined_segments = combined_segments + segment

        position += 1
        progress_bar.progress(position/len(segments))

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
    add_samplecharacteristics_xml_element(_format, '32', '24000')

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
    add_samplecharacteristics_xml_element(audio, '32', '24000')


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
                   
def tranlate_vtt_file(file, file_name, api_key, api_region, language, voice, remove_overlaps, use_existing_translations):

    #create results directory
    output_folder = "./results"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_reading_info = st.info('Reading VTT file:')
    texts, start_times, end_times = parse_vtt_file(file)
    st.success('Done!')

    translations_folder = os.path.join(output_folder, 'individual_audio_files') 
    if use_existing_translations:
        st.info("Loading translations from folder: " + translations_folder)
        translations = load_translations_from_folder(os.path.join(translations_folder, 'original'))
        st.success('Done!')
    else:
        st.info('Translating text to speach:')
        translations = translate_and_save_text(texts, destination=translations_folder, api_key=api_key, api_region=api_region, language=language, voice=voice)
        st.success('Done!')

    #Convert file paths to AudioSegments
    segments_original = load_audio_segments_from_files(translations, trim_end_ms=800) #Microsoft translation files have unwanted 800ms of silence at the end

    #Detect files overlap
    overlap_info = st.info('Checking if new translations will not overlap each other')
    segments_adjusted, overlaps = check_for_overlaps(segments_original, start_times, auto_shrink=remove_overlaps, allowed_overlap=50)
    st.success('Done!')

    if overlaps > 0 and not remove_overlaps:
        overlap_warning = st.warning('[WARNING] There are " + str(overlaps) + " overlap(s) in your files. Fix them manually by rewriting translation text or use --auto_remove_overlap flag to automatically remove overlays by speeding the translation file.')
    else:
        #combine files into one wave
        cobine_info = st.info('Combining audio files into a single file:')

        combined_segments = combine_segments(segments_adjusted, start_times)
        # combined_segments_file_path = os.path.join(output_folder, str(file_name)[:-3]+'wav') 
        combined_segments_file_path = os.path.join(output_folder,'result.wav') 
        combined_segments.export(combined_segments_file_path, format="wav")

        save_adjusted_translations(segments_adjusted, translations_folder)
        show_wave_visualization(combined_segments_file_path)
        generate_Adobe_Audition_FCP_XML(segments_adjusted, audio_folder=os.path.join(translations_folder, 'adjusted'),  start_times=start_times, file_name=os.path.join(output_folder, "adobe_audition_output_adjusted.xml"))
        
        st.success('Done!')

    generate_Adobe_Audition_FCP_XML(segments_original, audio_folder=os.path.join(translations_folder, 'original'),  start_times=start_times, file_name=os.path.join(output_folder, "adobe_audition_original.xml"))

    zip_directory('./result', './results')

    # st.download_button('Download file', binary_contents)  # Defaults to 'application/octet-stream'

    with open('result.zip', 'rb') as f:
        st.download_button('Download translations', f, file_name='result.zip')  # Defaults to 'application/octet-stream'


    # with open('./result.zip') as f:
    #     st.download_button('Download results', f)

######################
# UI STREAMLIT METHODS
######################

@st.cache
def get_regions_DataFrame():
    regions = {'Geography': ['Africa', 'Asia Pacific', 'Asia Pacific', 'Asia Pacific', 'Asia Pacific', 'Asia Pacific', 'Asia Pacific', 'Asia Pacific', 'Canada', 'Europe', 'Europe', 'Europe', 'Europe', 'Europe', 'Europe', 'Europe', 'Europe', 'Middle East', 'South America', 'US', 'US', 'US', 'US', 'US', 'US', 'US', 'US', 'US'],
    'Region':['South Africa North', 'East Asia', 'Southeast Asia', 'Australia East', 'Central India', 'Japan East', 'Japan West', 'Korea Central', 'Canada Central', 'North Europe', 'West Europe', 'France Central', 'Germany West Central', 'Norway East', 'Switzerland North', 'Switzerland West', 'UK South', 'UAE North', 'Brazil South', 'Central US', 'East US', 'East US 2', 'North Central US', 'South Central US', 'West Central US', 'West US', 'West US 2', 'West US 3'],
    'Region identifier':['southafricanorth', 'eastasia', 'southeastasia', 'australiaeast', 'centralindia', 'japaneast', 'japanwest', 'koreacentral', 'canadacentral', 'northeurope', 'westeurope', 'francecentral', 'germanywestcentral', 'norwayeast', 'switzerlandnorth', 'switzerlandwest', 'uksouth', 'uaenorth', 'brazilsouth', 'centralus', 'eastus', 'eastus2', 'northcentralus', 'southcentralus', 'westcentralus', 'westus', 'westus2', 'westus3']}

    return pd.DataFrame(regions)

import requests

@st.cache
def get_available_languages_for_region(region, api_key):

    try:  
        url = 'https://' + region + '.tts.speech.microsoft.com/cognitiveservices/voices/list'
        headers = {'Ocp-Apim-Subscription-Key':api_key}
        auth = requests.auth.HTTPBasicAuth('apikey', api_key)
        r = requests.get(url, auth=auth, headers=headers)

        return pd.DataFrame.from_records(r.json())

    except:
        return None


######################
# ZIP METHODS
######################       
# import io
# import zipfile

# files array format [('file_name1.ext', io.BytesIO), ('file_name1.ext', io.BytesIO)]
# def zip_files(zip_name, files):

#     zip_buffer = io.BytesIO()
#     with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
#         for file_name, data in files:
#             zip_file.writestr(file_name, data.getvalue())

#     with open(zip_name, 'wb') as f:
#         f.write(zip_buffer.getvalue())

import shutil
def zip_directory(zip_name, path):
    shutil.make_archive(zip_name,'zip', path)
    

def main():
    global args

    debug = False

    args = parse_args()
    accepted_file_types = [".vtt"]

    st.markdown('## Convert VTT to speech')
    st.sidebar.title('Settings')
    
    api_key = st.sidebar.text_input("API key: (required)")
    if api_key == '': api_key_info = st.warning('Please add your [Azure API](https://portal.azure.com/#home) key in the sidebar.')
    st.sidebar.markdown('---')

    regions = get_regions_DataFrame()
    if debug:
        show_regions = st.sidebar.checkbox('Show available regions')
        if show_regions:
            st.sidebar.table(regions)
    api_geography = st.sidebar.selectbox("Pick Azure service geography: (required)", regions['Geography'].unique())
    api_region = st.sidebar.selectbox("Pick Azure service region: (required)", regions.loc[regions['Geography']== api_geography, 'Region'])   
    api_region_id = regions.loc[regions['Region'] == api_region, 'Region identifier'].iloc[0]
    if debug: st.sidebar.write("API region id: ", api_region_id)

    languages_data = get_available_languages_for_region(api_region_id, api_key)
    
    if languages_data is not None:
        language = st.sidebar.selectbox("Pick language:", languages_data['LocaleName'].unique())
        language_locale = languages_data.loc[languages_data['LocaleName'] == language, 'Locale'].iloc[0]
        if debug: st.sidebar.write("Locale: ", language_locale)

        gender = st.sidebar.selectbox("Pick voice gender: (required)", ["Female", "Male"])

        voices = languages_data.loc[(languages_data['LocaleName'] == language) & (languages_data['Gender'] == gender), ['DisplayName', 'ShortName']]
        voice = st.sidebar.selectbox("Pick voice: (required)", voices['DisplayName'])
        voice_api_name = voices.loc[voices['DisplayName'] == voice, 'ShortName'].iloc[0]
        if debug:  st.sidebar.write("Voice api name: ", voice_api_name)

        remove_overlaps = st.sidebar.checkbox('Automatically remove overlaps')
        use_existing_translations = st.sidebar.checkbox('Use existing translations (for debuging)')

        vtt_file = st.file_uploader(label="Upload VTT file", type=['vtt'], accept_multiple_files=False)
        if vtt_file is not None:
            # bytes_data = vtt_file.read()
            stringio = StringIO(vtt_file.getvalue().decode("utf-8"))
            string_data = stringio.read()

            tranlate_vtt_file(file=string_data, file_name=vtt_file.name, api_key=api_key, api_region=api_region_id, language=language_locale, voice=voice_api_name, remove_overlaps=remove_overlaps, use_existing_translations=use_existing_translations)
            st.balloons()
    else:
        st.error('**[ERROR]** Could not load available translation types for region **'+ api_region +'**.\n\n Possible reasons:\n* You don\'t have internet \n*  Your API key does not match the **' + api_region + '** region. Try choosing a different region.')

if __name__ == "__main__":
	main()