# Clean the source data

import re
import string
from dateutil import parser
from rmgarbage import Rmgarbage

def remove_headers(document, verbose=False):

    # eg 2003 U.S. Dist. Ct. Pleadings 3030, *5; 2004 U.S. Dist. Ct. Pleadings LEXIS 9181, **8
    # eg U.S. ex rel. Camillo v. Ancilla Systems, Inc., 233 F.R.D. 520 (2005)

    lines = document.split('\n')
    new_lines = []
    for line in lines:
        # digits
        line = re.sub('[0-9]', '@', line).strip()
        words = line.split(' ')
        stripped_line = []
        for word in words:
            # remove characters preceded by *
            if word != '':
                if not word[0] == '*':
                    stripped_line.append(word)
        new_lines.append(' '.join(stripped_line))

    # if stripped line occurs multiple times, remove in original
    headers_removed = []
    for i, line in enumerate(lines):
        # for line in original document
        stripped_line = new_lines[i]
        if new_lines.count(stripped_line) == 1 or len(stripped_line) < 20 or (not any(char == '@' for char in stripped_line) and not re.search('http', stripped_line)):
            # len is heuristic to stop it removing 'and', 'v' etc, sim for isalpha
            headers_removed.append(line)
        else:
            if verbose:
                print(stripped_line)

    no_headers_doc = '\n'.join(headers_removed)
    return no_headers_doc


def dirty(line, verbose=False):
    # Dirty lines that just have to be removed

    # PAGE NUMBERS
    # Format eg Page 5
    if re.match('Page [0-9]+$', line.strip()):  # match = anchored at beginning of strng
        if verbose:
            print('PAGE CASE A: '  + line)
        return True
    if re.match('Page[0-9]+$', line.strip()):
        if verbose:
            print('PAGE CASE A-OCR: '  + line)
        return True
    # Format eg lines containing just a number
    if re.match('[0-9]+$', line.strip()):
        if verbose:
            print('PAGE CASE B: '  + line)
        return True
    # Format eg 1 of 16
    if re.match('[0-9]+ of [0-9]+$', line.strip()):
        if verbose:
            print('PAGE CASE C: '  + line)
        return True
    # Format eg Page 1 of 16
    if re.match('Page [0-9]+ of [0-9]+$', line.strip()):
        if verbose:
            print('PAGE CASE D: '  + line)
        return True
    # Sometimes part of a larger header
    if re.search('[0-9]+ of [0-9]+', line) and (re.search('page', line.lower()) or re.search('case:', line.lower())):
        if verbose:
            print('PAGE CASE E: '  + line)
        return True

    # eg contains link (begins http)
    # NOTE: SOMETIMES USEFUL TO HAVE A LINK EG VAL[3] BARBARI CASE
    if re.search('http', line) and re.search('dock', line.lower()):
        if verbose:
            print('LINK: '  + line)
        return True

    # JUNK
    # eg " R-,’:P ";9 that contain no words
    # eg -4 Honorable SWene~io.dr,e]Ul .AS..NDiilest~rict ,Judge - corrupted!
    garbage_detector = Rmgarbage()
    if garbage_detector.is_garbage(line):
        if verbose:
            print('JUNK: '  + line)
            print('code: ' + garbage_detector.is_garbage(line))
        return True

    # TIMESTAMPS
    # eg 5/30/2007 4:06 PM
    try:
        # dateutils parser handles most date formats but will throw an error
        # if there is non date information
        parser.parse(line)
        if verbose:
            print('TIMESTAMP: '  + line)
        return True
    except:
        pass

    # JUST NUMBERS
    if not any(char.isalpha() for char in line):
        if verbose:
            print('FLOATING NUMBERS: ' + line)
        return True

    # If it has passed all the tests, it is okay
    return False



def clean(line, verbose=False):  # done!!!!
    # Cleaning the lines that passed the filtering

    # Remove ¶ and § - artefacts from scanning or document format
    line = line.replace('¶', '')
    line = line.replace('§\n', '')
    line = line.replace(' .', '.')

    if verbose:
        print('replaced weird symbols')

    words = line.split(' ')

    words_new = []
    for word in words:
        # Remove rest of paragraph / line after a numeral attached to a letter
        if re.match('[0-9][A-Za-z]+', word):
            if verbose:
                print('removed footnote')
            break

        # Remove references of form [**1], [*1]
        if len(word) > 1:
            if word[0] == '[' and word[1] == '*' and word[-1] == ']':
                continue

        # Remove (strings of) punctuation not connected to a letter
        word_fs = word.replace('.', 'A')  # deals with keeping isolated full stop
        punc_stripped = word_fs.translate(str.maketrans('', '', string.punctuation))
        if not punc_stripped == '' or punc_stripped == '§':
            words_new.append(word)
        else:
            if verbose:
                print('removed trailing punctuation')

    return ' '.join(words_new)


def line_breaks(doc, docket=False):

    # remove nonprintable chars messing stuff up
    #doc = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', doc)
    doc = re.sub(r'\x0c', '', doc)

    # replace all blank lines / strings of newline with one
    doc = re.sub(r'\n+', '\n', doc).strip()

    # colon line break if not already (need space so not in links)
    doc = re.sub(r': ([A-Za-z]\.)', r':\n\1', doc)

    # insert a new line before new legal paragraph (starts with number or letter in lists)
    doc = re.sub(r' ([0-9]+\. )', r'\n\1', doc)
    doc = re.sub(r' ([0-9]+\. )', r'\n\1', doc)  # not sure why this is necessary but it is
    doc = re.sub(r'\. ([A-Za-z]+\. )', r'\n\1', doc)

    if not docket:

        # Keep newline only if last character is : or . and following subheadings
        lines = doc.split('\n')
        new_lines = []
        for line in lines:
            if line == line.upper():
                line = 'PLACEHOLDER-H' + line + 'PLACEHOLDER-H'
            new_lines.append(line)
        doc = '\n'.join(new_lines)

        doc = doc.replace(':\n', 'PLACEHOLDER-C')
        doc = doc.replace('.\n', 'PLACEHOLDER-FS')
        doc = doc.replace('\n', ' ')
        doc = doc.replace('PLACEHOLDER-C', ':\n')
        doc = doc.replace('PLACEHOLDER-FS', '.\n')
        doc = doc.replace('PLACEHOLDER-H', '\n')


        # Remove the newline if the . is due to an abbreviation eg v. or U.S. (caps?)
        doc = doc.replace('v.\n', 'v. ')  # special case - versus
        for letter in string.ascii_uppercase:
            input = letter + '.\n'
            output = letter + '. '
        doc = re.sub(input, output, doc)


    # replace all blank lines / strings of newline with one
    doc = re.sub(r'\n+', '\n', doc).strip()

    # semicolon for list seps
    doc = re.sub(r'; ([A-Za-z]\.)', r';\n\1', doc)
    #doc = doc.replace(';', ';\n').strip()
    doc = doc.replace('\n ', '\n')

    # colon line break if not already (need space so not in links)
    doc = re.sub(r': ([A-Za-z]\.)', r':\n\1', doc)

    # insert a new line before new legal paragraph (starts with number or letter in lists)
    doc = re.sub(r' ([0-9]+\. )', r'\n\1', doc)
    doc = re.sub(r' ([0-9]+\. )', r'\n\1', doc)  # not sure why this is necessary but it is
    doc = re.sub(r'\. ([A-Za-z]+\. )', r'\n\1', doc)

    lines = doc.split('\n')
    new_lines = []
    for line in lines:
        if line.strip():
            new_lines.append(line)
    doc = '\n'.join(new_lines)

    return doc


def docket_processing(doc, verbose=False):
    # Remove lines that are just dates or beginning 'Date Filed'

    lines = doc.split('\n')
    new_lines = []
    for line in lines:
        # Remove numbers at start of lines
        line = line.lstrip('0123456789 ')

        # Remove lines that begin date filed AND NO TEXT AFTER
        if line.lower().startswith('date filed'):
            # Everything after date filed
            last_part = line.lower().split('date filed')[1]
            if all(x in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '/', ' '] for x in last_part):
                continue

        # Remove lines that are just dates
        words = line.split(' ')
        non_dates = False
        for word in words:
            try:
                # dateutils parser handles most date formats but will throw an error
                # if there is non date information
                parser.parse(word)
            except:
                # accounts for OCR errors
                for char in word:
                    if char not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '/', ' ']:
                        non_dates = True
        if non_dates is False:
            continue

        new_lines.append(line)

    return '\n'.join(new_lines)


def clean_document(doc, verbose=False):
    # Remove all lines following a footer marker
    if verbose:
        print('Removing footers...')
    # If line contains 'Print Completed' eg ********** Print Completed **********
    doc = doc.split('Print Completed')[0]
    # If line contains PACER Service Center
    doc = doc.split('PACER Service Center')[0]

    # Before line breaks
    doc = doc.replace('§\n', '\n')

    # Remove headers
    if verbose:
        print('Removing headers...')
    doc = remove_headers(doc, verbose=verbose)

    # Consider lines
    lines = doc.split('\n')

    # Remove lines which are dirty
    if verbose:
        print('Removing dirty lines...')
    lines = [line for line in lines if not dirty(line, verbose=verbose)]

    # Sort line breaks
    if verbose:
        print('Sorting line breaks...')
    doc = line_breaks(doc)

    # Clean remaining lines
    if verbose:
        print('Cleaning remaining lines...')
    lines = [clean(line, verbose=verbose) for line in lines]
    doc = '\n'.join(lines)

    # Find if document is docket (is 'CIVIL DOCKET' in text?)
    if 'CIVIL DOCKET' in doc:
        if verbose:
            print('Document is a docket... processing more...')
        doc = docket_processing(doc, verbose=verbose)
        if verbose:
            print('Checking line breaks...')
        doc = line_breaks(doc, docket=True)

    else:
        # Line breaks again
        if verbose:
            print('Checking line breaks...')
        doc = line_breaks(doc)

    return doc

# #For testing!!!!
# print('Reading file...')
# with open('train0_0.txt', 'r') as f:
#      doc = f.read()
#
# doc = clean_document(doc, verbose=True)
#
# # Output
# print('Outputting clean file...')
# with open('output.txt', 'w') as f:
#     f.write(doc)
