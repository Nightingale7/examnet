class Problem:

    def __init__(self, sut):
        self.sut = sut

    @classmethod
    def from_revision(C, revision):
        return C(revision.get_sut())

    def get_description(self):
        return self.__class__.__name__

    def get_sut(self):
        return self.sut

    def get_objectives(self):
        raise NotImplementedError

