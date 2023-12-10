import numpy as np
import time
import random
from BitVector import *

Sbox = (
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
)

InvSbox = (
    0x52, 0x09, 0x6A, 0xD5, 0x30, 0x36, 0xA5, 0x38, 0xBF, 0x40, 0xA3, 0x9E, 0x81, 0xF3, 0xD7, 0xFB,
    0x7C, 0xE3, 0x39, 0x82, 0x9B, 0x2F, 0xFF, 0x87, 0x34, 0x8E, 0x43, 0x44, 0xC4, 0xDE, 0xE9, 0xCB,
    0x54, 0x7B, 0x94, 0x32, 0xA6, 0xC2, 0x23, 0x3D, 0xEE, 0x4C, 0x95, 0x0B, 0x42, 0xFA, 0xC3, 0x4E,
    0x08, 0x2E, 0xA1, 0x66, 0x28, 0xD9, 0x24, 0xB2, 0x76, 0x5B, 0xA2, 0x49, 0x6D, 0x8B, 0xD1, 0x25,
    0x72, 0xF8, 0xF6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xD4, 0xA4, 0x5C, 0xCC, 0x5D, 0x65, 0xB6, 0x92,
    0x6C, 0x70, 0x48, 0x50, 0xFD, 0xED, 0xB9, 0xDA, 0x5E, 0x15, 0x46, 0x57, 0xA7, 0x8D, 0x9D, 0x84,
    0x90, 0xD8, 0xAB, 0x00, 0x8C, 0xBC, 0xD3, 0x0A, 0xF7, 0xE4, 0x58, 0x05, 0xB8, 0xB3, 0x45, 0x06,
    0xD0, 0x2C, 0x1E, 0x8F, 0xCA, 0x3F, 0x0F, 0x02, 0xC1, 0xAF, 0xBD, 0x03, 0x01, 0x13, 0x8A, 0x6B,
    0x3A, 0x91, 0x11, 0x41, 0x4F, 0x67, 0xDC, 0xEA, 0x97, 0xF2, 0xCF, 0xCE, 0xF0, 0xB4, 0xE6, 0x73,
    0x96, 0xAC, 0x74, 0x22, 0xE7, 0xAD, 0x35, 0x85, 0xE2, 0xF9, 0x37, 0xE8, 0x1C, 0x75, 0xDF, 0x6E,
    0x47, 0xF1, 0x1A, 0x71, 0x1D, 0x29, 0xC5, 0x89, 0x6F, 0xB7, 0x62, 0x0E, 0xAA, 0x18, 0xBE, 0x1B,
    0xFC, 0x56, 0x3E, 0x4B, 0xC6, 0xD2, 0x79, 0x20, 0x9A, 0xDB, 0xC0, 0xFE, 0x78, 0xCD, 0x5A, 0xF4,
    0x1F, 0xDD, 0xA8, 0x33, 0x88, 0x07, 0xC7, 0x31, 0xB1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xEC, 0x5F,
    0x60, 0x51, 0x7F, 0xA9, 0x19, 0xB5, 0x4A, 0x0D, 0x2D, 0xE5, 0x7A, 0x9F, 0x93, 0xC9, 0x9C, 0xEF,
    0xA0, 0xE0, 0x3B, 0x4D, 0xAE, 0x2A, 0xF5, 0xB0, 0xC8, 0xEB, 0xBB, 0x3C, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2B, 0x04, 0x7E, 0xBA, 0x77, 0xD6, 0x26, 0xE1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0C, 0x7D,
)

Mixer = [
    [BitVector(hexstring="02"), BitVector(hexstring="03"), BitVector(hexstring="01"), BitVector(hexstring="01")],
    [BitVector(hexstring="01"), BitVector(hexstring="02"), BitVector(hexstring="03"), BitVector(hexstring="01")],
    [BitVector(hexstring="01"), BitVector(hexstring="01"), BitVector(hexstring="02"), BitVector(hexstring="03")],
    [BitVector(hexstring="03"), BitVector(hexstring="01"), BitVector(hexstring="01"), BitVector(hexstring="02")]
]

InvMixer = [
    [BitVector(hexstring="0E"), BitVector(hexstring="0B"), BitVector(hexstring="0D"), BitVector(hexstring="09")],
    [BitVector(hexstring="09"), BitVector(hexstring="0E"), BitVector(hexstring="0B"), BitVector(hexstring="0D")],
    [BitVector(hexstring="0D"), BitVector(hexstring="09"), BitVector(hexstring="0E"), BitVector(hexstring="0B")],
    [BitVector(hexstring="0B"), BitVector(hexstring="0D"), BitVector(hexstring="09"), BitVector(hexstring="0E")]
]

AES_modulus = BitVector(bitstring='100011011')


def takeKeyInput():
    print('Key')
    str = input('In ASCII: ').encode('utf-8')
    if (len(str) < 16):
        str = str.ljust(16)
    if len(str) > 16:
        str = str[:16]
    hex_str = str.hex()
    print("In HEX: " + hex_str)
    array = [hex_str[i:i + 2] for i in range(0, len(hex_str), 2)]

    matrix = np.array(array).reshape(4, 4).T

    # print('At first the matrix')
    # print(matrix)

    return matrix


def takeMsgInput():
    print('Plain text')
    string = input('In ASCII: ').encode('utf-8')
    if len(string) < 16:
        string = string.ljust(16)
    if len(string) > 16:
        rem = len(string) % 16
        if rem != 0:
            needToAdd = 16 - rem
            string = string.ljust(len(string) + needToAdd)

    hex_str = string.hex()
    print("In HEX: " + hex_str)

    # print(len(hex_str))

    array = [hex_str[i:i + 2] for i in range(0, len(hex_str), 2)]
    array = np.array(array)
    totalBlk = len(string) / 16
    allBlks = np.array_split(array, totalBlk)
    allBlks = np.array(allBlks)
    # matrix = np.array(array).reshape(4, 4).T
    matrix = []
    for i in range(len(allBlks)):
        mat = allBlks[i]
        mat = np.array(mat).reshape(4, 4).T
        matrix.append(mat)
    matrix = np.array(matrix)
    # print(matrix)
    return matrix


def circShift(givenWord, n):
    shiftedWord = np.roll(givenWord, n)
    # print('left shift er por')
    # print(shiftedWord)
    return shiftedWord


def byteSub(givenWord):
    hexStr = ''
    for i in range(4):
        theString = givenWord[i]
        # print('the str is ')
        # print(theString)
        if len(theString) != 2:
            word1 = '0'
            word2 = theString[0]
        else:
            word1 = theString[0]
            word2 = theString[1]
        intVal1 = int(word1, 16)
        intVal2 = int(word2, 16)
        # print(intVal1)
        # print(intVal2)
        idx = intVal1 * 16 + intVal2
        # np.insert(arr, i, Sbox[idx])
        hex_value = hex(Sbox[idx])
        onlyDigit = hex_value.split('x')
        if len(onlyDigit[1]) != 2:
            onlyDigit[1] = '0' + onlyDigit[1]
        hexStr = hexStr + onlyDigit[1]
        # print('idx e ')
        # print(onlyDigit)
        # print(hexStr)
        # print()

    array = [hexStr[i:i + 2] for i in range(0, len(hexStr), 2)]
    # print('byte sub sheshe')
    # print(array)
    return array


roundConsti = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36]


def operationG(keyWord, i):
    gKeyWord = circShift(keyWord, -1)
    gKeyWord = byteSub(gKeyWord)
    hexValue = int(gKeyWord[0], 16)
    wholeHexVal = hex(hexValue ^ roundConsti[i])
    onlydigit = wholeHexVal.split('x')[1]
    if len(onlydigit) != 2:
        onlydigit = '0' + onlydigit
    gKeyWord[0] = onlydigit
    # print('operation g er pore')
    # print(gKeyWord)
    return gKeyWord


def xorWord(word1, word2):
    result = []

    for i in range(4):
        # if word1[i] == '0d' or word2[i] == '0d':
        #     # print()
        #     # print('eikhanneee')
        #     # print(word1, word2, sep="<<..>>")
        #     hexValue1 = int(word1[i], 16)
        #     hexValue2 = int(word2[i], 16)
        #     wholeHexXorVal = hex(hexValue1 ^ hexValue2)
        #     onlydigit = wholeHexXorVal.split('x')[1]
        #     # print(onlydigit)
        #     # print()
        #     # print()
        hexValue1 = int(word1[i], 16)
        hexValue2 = int(word2[i], 16)
        wholeHexXorVal = hex(hexValue1 ^ hexValue2)
        onlydigit = wholeHexXorVal.split('x')[1]
        if len(onlydigit) != 2:
            onlydigit = '0' + onlydigit
        result.append(onlydigit)

    return result


def keyGeneration(key0Matrix):
    keyArray = []
    keyArray.append(key0Matrix)
    currentKey = key0Matrix
    for i in range(10):
        # print('tomo op')
        # print(i)
        word = []
        word.append(currentKey[:, 0])
        word.append(currentKey[:, 1])
        word.append(currentKey[:, 2])
        word.append(currentKey[:, 3])
        gWord3 = operationG(word[3], i)
        keyMatrix = []
        word.append(xorWord(word[0], gWord3))
        keyMatrix.append(word[4])

        for j in range(1, 4):
            word.append(xorWord(word[(4 + j - 1)], word[j]))
            keyMatrix.append(word[4 + j])

        numpyKeyMatrix = np.array(keyMatrix)
        numpyKeyMatrix = numpyKeyMatrix.transpose()
        keyArray.append(numpyKeyMatrix)
        currentKey = numpyKeyMatrix
        # print('current key')
        # print(currentKey)

    keyNumpyArr = np.array(keyArray)
    return keyNumpyArr


def subByteRound(stateMatrix):
    currstate = []
    for i in range(4):
        word = stateMatrix[:, i]
        # print(word)
        word = byteSub(word)
        currstate.append(word)
    currstate = np.array(currstate)
    currstate = currstate.transpose()
    # print(currstate)
    return currstate


def shiftRowRound(stateMatrix):
    currstate = []
    for i in range(4):
        word = stateMatrix[i, :]
        # print(word)
        word = circShift(word, -i)
        currstate.append(word)
    currstate = np.array(currstate)
    # print(currstate)
    return currstate


def mixcolRound(stateMatrix):
    # print('state matrix ')
    # print(stateMatrix)
    currState = []
    for i in range(0, 4):
        onerow = []
        for j in range(0, 4):
            oneElement = 0
            for k in range(0, 4):
                # bv1 = BitVector(hexstring="02")
                # bv2 = BitVector(hexstring="63")
                # bv3 = bv1.gf_multiply_modular(bv2, AES_modulus, 8)
                hexValueState = int(stateMatrix[k][j], 16)
                hexValueState = hex(hexValueState)
                hexValueState = hexValueState.split('x')
                hexValueState = hexValueState[1]
                bv1 = BitVector(hexstring=hexValueState)
                mixColVal = Mixer[i][k]
                # mixColVal = mixColVal.int_val()
                # mixColVal = str(mixColVal)
                # mixColVal = int(mixColVal, 16)
                # print()
                # print('guun hoitese')
                # print(mixColVal, bv1, sep="<<..>>")
                guun = mixColVal.gf_multiply_modular(bv1, AES_modulus, 8)
                # print(guun)
                guun = guun.int_val()
                # print('guuner in val')
                # print(guun)
                guun = hex(guun)
                # print('guuner string val')
                # print(guun)
                guun = int(guun, 16)
                # print('gunner hex value')
                # print(guun)
                # print()
                # print('xor hoitese')
                # print(oneElement, guun, sep="<<..>>")

                oneElement = oneElement ^ guun
                # print(oneElement)

            oneElement = hex(oneElement)
            oneElement = oneElement.split('x')
            oneElement = oneElement[1]
            if len(oneElement) < 2:
                oneElement = '0' + oneElement
            onerow.append(oneElement)
        currState.append(onerow)
        # print()
        # print('shesh ekta round')
        # print()

    currState = np.array(currState)
    return currState


def addRoundKeyRound(stateMatrix, roundKey):
    currState = []
    for i in range(4):
        onerow = []
        for j in range(4):
            stateVal = int(stateMatrix[i][j], 16)
            keyVal = int(roundKey[i][j], 16)
            element = stateVal ^ keyVal
            element = hex(element)
            element = element.split('x')
            element = element[1]
            if len(element) < 2:
                element = '0' + element
            onerow.append(element)
        currState.append(onerow)
    currState = np.array(currState)
    return currState


def fetchHexFromMat(matrices):
    strArr = ''
    for i in range(len(matrices)):
        for j in range(4):
            for k in range(4):
                theVal = int(matrices[i][j][k], 16)
                theString = hex(theVal)
                theString = theString.split('x')
                theString = theString[1]
                if len(theString) < 2:
                    theString = '0' + theString
                strArr = strArr + theString
    # print(strArr)

    return strArr


def printHex(strArr):
    for i in range(len(strArr)):
        print(strArr[i], end='')
    print()


def printAsciiFromHex(strArr):
    # print()
    # print()
    # print('the string is')
    # print(strArr)
    # print()
    # print()
    # hexString = ''
    # for i in range(len(strArr)):
    #     theVal = int(strArr[i], 16)
    #     string = hex(theVal)
    #     string = string.split('x')
    #     string = string[1]
    #     if len(string) < 2:
    #         string = '0' + string
    #     hexString = hexString + string
    # byte_string = bytes.fromhex(hexString)

    ascii_string = ''.join([chr(int(strArr[i:i + 2], 16)) for i in range(0, len(strArr), 2)])
    print(ascii_string)


def invShiftRowRound(stateMatrix):
    currstate = []
    for i in range(4):
        word = stateMatrix[i, :]
        word = circShift(word, i)
        currstate.append(word)
    currstate = np.array(currstate)
    return currstate


def invByteSub(givenWord):
    hexStr = ''
    for i in range(4):
        theString = givenWord[i]
        if len(theString) != 2:
            word1 = '0'
            word2 = theString[0]
        else:
            word1 = theString[0]
            word2 = theString[1]
        intVal1 = int(word1, 16)
        intVal2 = int(word2, 16)
        idx = intVal1 * 16 + intVal2
        hex_value = hex(InvSbox[idx])
        onlyDigit = hex_value.split('x')
        if len(onlyDigit[1]) != 2:
            onlyDigit[1] = '0' + onlyDigit[1]
        hexStr = hexStr + onlyDigit[1]

    array = [hexStr[i:i + 2] for i in range(0, len(hexStr), 2)]
    return array


def invSubByteRound(stateMatrix):
    currstate = []
    for i in range(4):
        word = stateMatrix[:, i]
        word = invByteSub(word)
        currstate.append(word)
    currstate = np.array(currstate)
    currstate = currstate.transpose()
    return currstate


def invMixcolRound(stateMatrix):
    currState = []
    for i in range(0, 4):
        onerow = []
        for j in range(0, 4):
            oneElement = 0
            for k in range(0, 4):
                hexValueState = int(stateMatrix[k][j], 16)
                hexValueState = hex(hexValueState)
                hexValueState = hexValueState.split('x')
                hexValueState = hexValueState[1]
                bv1 = BitVector(hexstring=hexValueState)
                mixColVal = InvMixer[i][k]
                guun = mixColVal.gf_multiply_modular(bv1, AES_modulus, 8)
                guun = guun.int_val()
                guun = hex(guun)
                guun = int(guun, 16)
                oneElement = oneElement ^ guun
            oneElement = hex(oneElement)
            oneElement = oneElement.split('x')
            oneElement = oneElement[1]
            if len(oneElement) < 2:
                oneElement = '0' + oneElement
            onerow.append(oneElement)
        currState.append(onerow)

    currState = np.array(currState)
    return currState


# input plain text and key
key0Matrix = takeKeyInput()
plainTextMatrices = takeMsgInput()

# key generation
keyStartTime = time.time()
allKeys = keyGeneration(key0Matrix)
keyEndTime = time.time()
elapsedKeyTime = (keyEndTime - keyStartTime) * 1000

# encryption
# round0
encryptStartTime = time.time()
currentStateMatrices = []
for i in range(len(plainTextMatrices)):
    currState = addRoundKeyRound(plainTextMatrices[i], allKeys[0])
    currentStateMatrices.append(currState)
currentStateMatrices = np.array(currentStateMatrices)

# round1 to 10
for i in range(10):
    currStateMatrices = []
    if i != 9:
        # 1-9 round with mix col
        for j in range(len(currentStateMatrices)):
            oneMatrix = currentStateMatrices[j]
            oneMatrix = subByteRound(oneMatrix)
            oneMatrix = shiftRowRound(oneMatrix)
            oneMatrix = mixcolRound(oneMatrix)
            oneMatrix = addRoundKeyRound(oneMatrix, allKeys[i + 1])
            currStateMatrices.append(oneMatrix)
        currStateMatrices = np.array(currStateMatrices)
        currentStateMatrices = currStateMatrices

    else:
        # without mix col
        for j in range(len(currentStateMatrices)):
            oneMatrix = currentStateMatrices[j]
            oneMatrix = subByteRound(oneMatrix)
            oneMatrix = shiftRowRound(oneMatrix)
            oneMatrix = addRoundKeyRound(oneMatrix, allKeys[i + 1])
            currStateMatrices.append(oneMatrix)
        currStateMatrices = np.array(currStateMatrices)
        currentStateMatrices = currStateMatrices

encryptEndTime = time.time()
elapsedEncryptTime = (encryptEndTime - encryptStartTime) * 1000

# ciphered text
cipheredMatrix = currentStateMatrices
ciMat = []
for i in range(len(cipheredMatrix)):
    matTrans = np.transpose(cipheredMatrix[i])
    ciMat.append(matTrans)
ciMat = np.array(ciMat)
print('Ciphered text')
hexString = fetchHexFromMat(ciMat)
print('In HEX: ', end=' ')
printHex(hexString)
print('In ASCII: ', end=' ')
printAsciiFromHex(hexString)

# Decryption
# round0
decStartTime = time.time()
currentMatrices = []
# print('Decryption er time e mat')
# print(currentStateMatrices)
#
# currSMataa = []
# for i in range(len(currentStateMatrices)):
#     currrrr = currentStateMatrices[i]
#     currrrr = np.transpose(currrrr)
#     currSMataa.append(currrrr)
# currStateMatrices = np.array(currSMataa)

for i in range(len(currentStateMatrices)):
    currState = addRoundKeyRound(currentStateMatrices[i], allKeys[10])
    currentMatrices.append(currState)
currentMatrices = np.array(currentMatrices)

# round 1-10
for i in range(10):
    currStateMatrices = []
    if i != 9:
        # 1-9 round with mix col
        for j in range(len(currentMatrices)):
            oneMatrix = currentMatrices[j]
            oneMatrix = invShiftRowRound(oneMatrix)
            oneMatrix = invSubByteRound(oneMatrix)
            oneMatrix = addRoundKeyRound(oneMatrix, allKeys[10 - (i + 1)])
            oneMatrix = invMixcolRound(oneMatrix)
            currStateMatrices.append(oneMatrix)
        currStateMatrices = np.array(currStateMatrices)
        currentMatrices = currStateMatrices

    else:
        # without mix col
        for j in range(len(currentMatrices)):
            oneMatrix = currentMatrices[j]
            oneMatrix = invShiftRowRound(oneMatrix)
            oneMatrix = invSubByteRound(oneMatrix)
            oneMatrix = addRoundKeyRound(oneMatrix, allKeys[10 - (i + 1)])
            currStateMatrices.append(oneMatrix)
        currStateMatrices = np.array(currStateMatrices)
        currentMatrices = currStateMatrices

decipheredMatrix = currentMatrices
mattNow = []
for i in range(len(decipheredMatrix)):
    matrixNow = np.transpose(decipheredMatrix[i])
    mattNow.append(matrixNow)
decipheredMatrix = np.array(mattNow)

decryptEndTime = time.time()
elapsedDecryptTime = (decryptEndTime - decStartTime) * 1000

# deciMat = []
# for i in range(len(decipheredMatrix)):
#     matTrans = np.transpose(decipheredMatrix[i])
#     deciMat.append(matTrans)
# deciMat = np.array(deciMat)
print('Deciphered text')
hexString = fetchHexFromMat(decipheredMatrix)
print('In HEX: ', end=' ')
printHex(hexString)
print('In ASCII: ', end=' ')
printAsciiFromHex(hexString)

# time printing
print('Execution Time Details')
print('Key Schedule Time: ', elapsedKeyTime, ' ms', sep=' ')
print('Encryption Time: ', elapsedEncryptTime, ' ms', sep=' ')
print('Decryption Time: ', elapsedDecryptTime, ' ms', sep=' ')


# key = [['d2', '15', '63', 'c3'],
#        ['60', '7a', '39', '03'],
#        ['0d', 'bc', 'e9', '1e'],
#        ['e7', '68', '01', 'fb']]
# key = np.array(key)
# gw3 = operationG(key[:, 3], 3)
# w4 = xorWord(key[:, 0], gw3)
# print(gw3)
# print(w4)
# arr = keyGeneration(key)
# print(arr)


# #key scheduling
# key0Mat = takeInput()
# allKeys = keyGeneration(key0Mat)
# print(allKeys)


# matrix = [
#     ['00', '3c', '6e', '47'],
#     ['1f', '4e', '22', '74'],
#     ['0e', '08', '1b', '31'],
#     ['54', '59', '0b', '1a']
# ]
#
# key = [
# ['e2', '91', 'b1', 'd6'],
#     ['32', '12', '59', '79'],
#     ['fc', '91', 'e4', 'a2'],
#     ['f1', '88', 'e6', '93']
# ]
#
# matrix = np.array(matrix)
# curr = subByteRound(matrix)
# curr = shiftRowRound(curr)
# curr = mixcolRound(curr)
# curr = addRoundKeyRound(curr, key)
# print(curr)

# b = BitVector(hexstring="10")
# int_val = b.intValue()
# mixColVal = Mixer[1][1]
# # mixColVal = int(mixColVal, 16)
# print(mixColVal.int_val())

# bv1 = BitVector(hexstring="02")
# bv2 = BitVector(hexstring="63")
# bv3 = bv1.gf_multiply_modular(bv2, AES_modulus, 8)
# print(bv3.int_val())

# byteSub(mat[:,3])
# arr = circShift(key0Mat[:, 3], -1)
# gkey = operationG(key0Mat[:, 3], 0)
# print(gkey)

# hex value parse and XOR
# hexValue = int(arr[0], 16)
# wholeHexVal = hex(hexValue ^ roundConsti[0])
# onlydigit = wholeHexVal.split('x')[1]
# arr[0] = onlydigit
# print(arr)


# arr[0] = arr[0] ^
# arr
# intVal = Sbox[0]
# hexStr = hex(intVal)
# print(arr)


# circLeftShift(mat, -1)



# diff hellmann
def primesInRange(x):
    primes = []
    for n in range(x):
        isPrime = True
        for num in range(2, n):
            if n % num == 0:
                isPrime = False
                break
        if isPrime:
            primes.append(n)
    return primes


primes = primesInRange(128)
thePrime = random.choice(primes)
list = []
for i in range(2, thePrime):
    list.append(i)
a = random.choice(list)
b = random.choice(list)
while (4 * (a ** 3) + 27 * (b ** 2)) % thePrime == 0:
    a = random.choice(list)
    b = random.choice(list)


def returnABP():
    return a, b, thePrime


def genPrivateKey():
    privateKey = thePrime
    while privateKey > thePrime-1:
        privateKey = random.getrandbits(128)
    return privateKey


# def calcy(x):
#     y = ((x**3) + (a*x) + b) % thePrime
#     if pow(y)
