"""
Sentence Builder module for constructing readable text from signs (words & letters)
"""
class SentenceBuilder:
    def __init__(self):
        self.words = []
        self.current_word = ""

    def process_sign(self, sign: str) -> str:
        """
        Processes an accepted sign (Word or Letter or Control gesture).
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
        elif len(sign) > 1:
            # Word-level sign recognized (e.g. "Hello", "Thank You", "Water", "Help")
            if self.current_word:
                self.words.append(self.current_word)
                self.current_word = ""
            self.words.append(sign)
        else:
            # Single letter recognized (e.g. "A", "B", "C")
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
            last_word = self.words.pop()
            if len(last_word) > 1:
                # Removed a full word sign
                pass
            else:
                self.current_word = last_word[:-1]

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
