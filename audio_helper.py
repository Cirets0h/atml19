import struct
# Adapted from https://stackoverflow.com/questions/4207326/parse-wav-file-header
def wav_info(filename):
    with open(filename, "rb") as fileIn:
        bufHeader = fileIn.read(38)
        # Verify that the correct identifiers are present
        #if (bufHeader[0:4] != "RIFF") or \
        #   (bufHeader[12:16] != "fmt "): 
        #     print("Input file not a standard WAV file")
        #     return
        # endif
        stHeaderFields = {'ChunkSize' : 0, 'Format' : '',
            'Subchunk1Size' : 0, 'AudioFormat' : 0,
            'NumChannels' : 0, 'SampleRate' : 0,
            'ByteRate' : 0, 'BlockAlign' : 0,
            'BitsPerSample' : 0, 'Filename': ''}
        # Parse fields
        stHeaderFields['ChunkSize'] = struct.unpack('<L', bufHeader[4:8])[0]
        stHeaderFields['Format'] = bufHeader[8:12]
        stHeaderFields['Subchunk1Size'] = struct.unpack('<L', bufHeader[16:20])[0]
        stHeaderFields['AudioFormat'] = struct.unpack('<H', bufHeader[20:22])[0]
        stHeaderFields['NumChannels'] = struct.unpack('<H', bufHeader[22:24])[0]
        stHeaderFields['SampleRate'] = struct.unpack('<L', bufHeader[24:28])[0]
        stHeaderFields['ByteRate'] = struct.unpack('<L', bufHeader[28:32])[0]
        stHeaderFields['BlockAlign'] = struct.unpack('<H', bufHeader[32:34])[0]
        stHeaderFields['BitsPerSample'] = struct.unpack('<H', bufHeader[34:36])[0]
    return stHeaderFields
