from utils.file_read_write import load_file

LINE_COMENT_START = '//'
LINE_BREAK = 'Ċ'
PARAGRAPH_COMMENT_START = '/*'
DOC_START = '/**'
DOC_END = '*/'


def cut_method(tokens, size, minimalNumberOfItemsAfterItem, item_to_keep):

    assert tokens is not None and len(tokens) >= size, "Original list must be bigger or of the same size as size."
    assert minimalNumberOfItemsAfterItem <= size / 2, "minimalNumberOfItemsAfterItem must be less than size."
    # list already of size
    if len(tokens) == size:
        return tokens
    # // list bigger.
    listSize: int = len(tokens)
    if item_to_keep is not None:
        itemIndex: int = tokens.index(item_to_keep)
        itemsAfterCount: int = listSize - itemIndex - 1
        minimalItemsAfterCount: int = min(minimalNumberOfItemsAfterItem, itemsAfterCount)
        maximumItemsBeforeCount: int = size - minimalItemsAfterCount - 1
        startIndex: int = max(0, itemIndex - maximumItemsBeforeCount)
    else:
        startIndex: int = listSize - size
    return startIndex, tokens[startIndex: size + startIndex]

def cut_method_codellama(tokens, size, minimalNumberOfItemsAfterItem, items_to_keep):
    #codeLlama tokenizer removes the masking token from the tokenized method and adds 3 other special tokens
    assert tokens is not None and len(tokens) >= size, "Original list must be bigger or of the same size as size."
    assert minimalNumberOfItemsAfterItem <= size / 2, "minimalNumberOfItemsAfterItem must be less than size."
    # list already of size
    if len(tokens) == size:
        return tokens
    # // list bigger.
    listSize: int = len(tokens)

    #if there are special tokens : the tokens always start with the token <_PRE>
    if tokens[0] in items_to_keep:
        size = size - 2
        sufIndex : int = tokens.index(items_to_keep[1])
        itemsAfterCount: int = listSize - sufIndex - 1
        minimalItemsAfterCount: int = min(minimalNumberOfItemsAfterItem, itemsAfterCount)
        maximumItemsBeforeCount: int = size - minimalItemsAfterCount - 1
        startIndex: int = max(0, sufIndex - maximumItemsBeforeCount)

        subList: list = tokens[startIndex:startIndex + size]
        if subList[0] != items_to_keep[0]:
            subList = [items_to_keep[0]] + subList
        if subList[len(subList) - 1] != items_to_keep[2]:
            subList.append(items_to_keep[2])
        return max(0,startIndex - 1), subList

    else:
        startIndex: int = listSize - size
        subList: list = tokens[startIndex:startIndex + size]
        return startIndex, subList

    assert tokens is not None and len(tokens) >= size, "Original list must be bigger or of the same size as size."
    assert minimalNumberOfItemsAfterItem <= size / 2, "minimalNumberOfItemsAfterItem must be less than size."
    # list already of size
    if len(tokens) == size:
        return tokens
    # // list bigger.
    listSize: int = len(tokens)
    if item_to_keep is not None:
        itemIndex: int = tokens.index(item_to_keep)
        itemsAfterCount: int = listSize - itemIndex - 1
        minimalItemsAfterCount: int = min(minimalNumberOfItemsAfterItem, itemsAfterCount)
        maximumItemsBeforeCount: int = size - minimalItemsAfterCount - 1
        startIndex: int = max(0, itemIndex - maximumItemsBeforeCount)
    else:
        startIndex: int = listSize - size
    return startIndex, tokens[startIndex: size + startIndex]

def surround_method(methodTokens, tokensBefore, tokensAfter, maximumTokensCount):
    tokensSize: int = len(methodTokens)
    missingTokens = maximumTokensCount - tokensSize
    if len(tokensBefore) < missingTokens / 2 - 1:
        result = tokensBefore + methodTokens
        if len(tokensAfter) > 0:
            tokensAfter = tokensAfter[0: min(maximumTokensCount - len(result), len(tokensAfter) - 1)]
            result += tokensAfter
    elif len(tokensAfter) < missingTokens / 2:
        maximumItemsBeforeCount = missingTokens - len(tokensAfter)
        result = []
        if len(tokensBefore) > 0:
            startIndex = max(0, len(tokensBefore) - maximumItemsBeforeCount)
            tokensBefore = tokensBefore[startIndex: len(tokensBefore) - 1]
            result += tokensBefore
        result += methodTokens
        result += tokensAfter
    else:
        if len(tokensAfter) > 0:
            maximumItemsBeforeCount = 1 + missingTokens / 2
        else:
            maximumItemsBeforeCount = missingTokens

        result = []
        if len(tokensBefore) > 0:
            startIndex = max(0, len(tokensBefore) - maximumItemsBeforeCount)
            tokensBefore = tokensBefore[int(startIndex): len(tokensBefore) - 1]
            result += tokensBefore

        result += methodTokens
        if len(tokensAfter) > 0:
            tokensAfter = tokensAfter[0: min(maximumTokensCount - len(result), len(tokensAfter) - 1)]
            result += tokensAfter

    assert len(result) <= maximumTokensCount
    assert len(result) == tokensSize + len(tokensBefore) + len(tokensAfter)
    if len(tokensBefore) == 0 or len(tokensAfter) == 0:
        print('empty')
    return result, tokensBefore, tokensAfter


class FileSnippetFittingOutput:
    def __init__(self, original_len, snippet_tokens, before_tokens, after_tokens, start_cutting_index):
        self.original_len = original_len
        self.snippet_tokens = snippet_tokens
        self.before_tokens = before_tokens
        self.after_tokens = after_tokens
        self.start_cutting_index = start_cutting_index


class FileSnippet:

    def __init__(self, file_path: str, start_char: int, end_char: int):
        self.file_path = file_path
        self.start_char = start_char
        self.end_char = end_char

    def load(self, file_string=None):
        if file_string is None:
            file_string = load_file(self.file_path)
        return file_string[self.start_char: self.end_char + 1]

    def tokenize(self, cbm, file_string=None):
        return cbm.tokenize(self.load(file_string))

    def fit_max(self, cbm, max_tokens, item_to_keep, file_string=None) -> FileSnippetFittingOutput:
        if file_string is None:
            file_string = load_file(self.file_path)
        snippet_tokens = self.tokenize(cbm, file_string)
        len_tokens = len(snippet_tokens)
        start_cutting_index = -1
        before_tokens = None
        after_tokens = None
        if len_tokens < max_tokens:
            max_tokens_to_add = max_tokens - len_tokens
            method_before_str = file_string[max(0, self.start_char - max_tokens_to_add):self.start_char - 1]
            method_after_str = file_string[
                               self.end_char + 1:min(self.end_char + 1 + max_tokens_to_add, len(file_string) - 1)]
            before_tokens = [] if len(method_before_str.strip()) == 0 else cbm.tokenize(method_before_str)
            after_tokens = [] if len(method_after_str.strip()) == 0 else cbm.tokenize(method_after_str)
            snippet_tokens, before_tokens, after_tokens = surround_method(snippet_tokens, before_tokens, after_tokens,
                                                                          max_tokens)
        elif len_tokens > max_tokens:
            start_cutting_index, snippet_tokens = cut_method(snippet_tokens, max_tokens, int(max_tokens / 3),
                                                             item_to_keep=item_to_keep)
        return FileSnippetFittingOutput(len_tokens, snippet_tokens, before_tokens, after_tokens, start_cutting_index)
