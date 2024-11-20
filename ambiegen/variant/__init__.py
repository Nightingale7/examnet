import heapq, os, random, shutil, sys, tempfile
from time import sleep

class Revision:

    def __init__(self, sut):
        self.sut = sut

    def get_sut(self):
        return self.sut

class RevisionSingleFile(Revision):
    """A revision such that its mutation means mutating a single Python source
    code file. This class performs the mutation using the mutmut library."""

    def __init__(self, sut, source_file, module_string=None, remove_files_on_unload=False):
        # source_file: Path to the revisions source code relative to current working directory.
        # module_string: The key corresponding to the revision module in sys.modules.
        # remove_files_on_unload: If true, then delete the source file when the class is unloaded.
        super().__init__(sut)
        self.source_file = source_file
        self.module_string = module_string
        self.remove_files_on_unload = remove_files_on_unload

        self.variant_path = "variant"

    def __del__(self):
        # Unload the module.
        if self.module_string is not None and self.module_string in sys.modules.keys():
            del sys.modules[self.module_string]

        # Remove source file and cache files if requested.
        if self.remove_files_on_unload:
            os.remove(self.source_file)
            path = os.path.join(os.path.dirname(self.source_file), "__pycache__")
            for file in os.listdir(path):
                if file.startswith(os.path.basename(self.source_file)[:-3]):
                    os.remove(os.path.join(path, file))

    def save_to_file(self, file_name):
        shutil.copy(self.source_file, file_name)


class Variant:

    def __init__(self, revisions):
        self.revisions = revisions

    @staticmethod
    def _get_variant_path():
        raise NotImplementedError

    @staticmethod
    def get_revision_class():
        raise NotImplementedError

    @staticmethod
    def get_original_revision():
        """Returns the original unmodified revision which is always the same no
        matter the variant."""

        pass

    @classmethod
    def load(C, identifier):
        path = os.path.join(C._get_variant_path(), identifier)
        revision_heap = []
        if not os.path.exists(path):
            return C([])

        for x in os.listdir(path):
            if not x.startswith("revision_"): continue
            idx = int(x.split(".")[0].split("_")[1])
            source_file = os.path.join(path, x)
            revision = C.get_revision_class().from_source_file(source_file)
            heapq.heappush(revision_heap, (idx, revision))

        return C([revision for _, revision in sorted(revision_heap)])

    def save(self, identifier, start_index=1, count=1):
        # Create a directory for the variant.
        # TODO: Check if already exists.
        path = os.path.join(self._get_variant_path(), identifier)
        os.makedirs(path, exist_ok=True)
        for n, revision in enumerate(self.revisions):
            count -= 1
            if count < 0:
                break
            target_file = os.path.join(path, "revision_{}.py".format(n+start_index))
            self.revisions[n].save_to_file(target_file)
            

    @property
    def n_revisions(self):
        return len(self.revisions)

    def get_revision(self, idx):
        return self.revisions[idx]

