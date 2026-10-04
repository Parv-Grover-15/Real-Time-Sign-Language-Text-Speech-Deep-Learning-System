"""
Sentence Builder module for constructing readable text from signs
"""
class SentenceBuilder:
    def __init__(self):
        self.raw_letters = []
        self.words = []
        self.current_word = ""

    def process_sign(self, sign: str) -> str:
        """
        Processes an accepted sign (letter, Space, Delete, or Clear).
        Updates sentence buffer and returns current sentence string.
        """
        if not sign:
            return self.get_text()

        sign_upper = sign.upper()

        if sign_upper == "SPACE" or sign == " ":
            self.add_space()
        elif sign_upper == "DELETE" or sign_upper == "BACKSPACE":
            self.delete_last()
        elif sign_upper == "CLEAR":
            self.clear()
        else:
            # Append character to current word
            self.current_word += sign

        return self.get_text()

    def add_space(self):
        if self.current_word:
            self.words.append(self.current_word)
            self.current_word = ""

    def delete_last(self):
        if self.current_word:
            self.current_word = self.current_word[:-1]
        elif self.words:
            self.current_word = self.words.pop()
            self.current_word = self.current_word[:-1]

    def clear(self):
        self.words = []
        self.current_word = ""

    def get_words(self) -> list:
        all_words = list(self.words)
        if self.current_word:
            all_words.append(self.current_word)
        return all_words

    def get_text(self) -> str:
        return " ".join(self.get_words())
