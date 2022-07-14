# Copyright 2022 Quantapix Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import hashlib
import os

import datasets


logger = datasets.logging.get_logger(__name__)

_DL_URLS = {
    # pylint: disable=line-too-long
    "cnn_stories": "https://drive.google.com/uc?export=download&id=0BwmD_VLjROrfTHk4NFg2SndKcjQ",
    "dm_stories": "https://drive.google.com/uc?export=download&id=0BwmD_VLjROrfM1BxdkxVaTY2bWs",
    "test_urls": "https://raw.githubusercontent.com/abisee/cnn-dailymail/master/url_lists/all_test.txt",
    "train_urls": "https://raw.githubusercontent.com/abisee/cnn-dailymail/master/url_lists/all_train.txt",
    "val_urls": "https://raw.githubusercontent.com/abisee/cnn-dailymail/master/url_lists/all_val.txt",
    # pylint: enable=line-too-long
}

_HIGHLIGHTS = "highlights"
_ARTICLE = "article"

_SUPPORTED_VERSIONS = [
    # Using cased version.
    datasets.Version("3.0.0", "Using cased version."),
    # Same data as 0.0.2
    datasets.Version("1.0.0", ""),
    # Having the model predict newline separators makes it easier to evaluate
    # using summary-level ROUGE.
    datasets.Version("2.0.0", "Separate target sentences with newline."),
]


_DEFAULT_VERSION = datasets.Version("3.0.0", "Using cased version.")


class CnnDailymailConfig(datasets.BuilderConfig):
    def __init__(self, **kw):
        super(CnnDailymailConfig, self).__init__(**kw)


def _get_url_hashes(path):
    """Get hashes of urls in file."""
    urls = _read_text_file(path)

    def url_hash(u):
        h = hashlib.sha1()
        try:
            u = u.encode("utf-8")
        except UnicodeDecodeError:
            logger.error("Cannot hash url: %s", u)
        h.update(u)
        return h.hexdigest()

    return {url_hash(u): True for u in urls}


def _get_hash_from_path(p):
    """Extract hash from path."""
    basename = os.path.basename(p)
    return basename[0 : basename.find(".story")]


def _find_files(dl_paths, publisher, url_dict):
    """Find files corresponding to urls."""
    if publisher == "cnn":
        top_dir = os.path.join(dl_paths["cnn_stories"], "cnn", "stories")
    elif publisher == "dm":
        top_dir = os.path.join(dl_paths["dm_stories"], "dailymail", "stories")
    else:
        logger.fatal("Unsupported publisher: %s", publisher)
    files = sorted(os.listdir(top_dir))

    ret_files = []
    for p in files:
        if _get_hash_from_path(p) in url_dict:
            ret_files.append(os.path.join(top_dir, p))
    return ret_files


def _subset_filenames(dl_paths, split):
    """Get filenames for a particular split."""
    assert isinstance(dl_paths, dict), dl_paths
    # Get filenames for a split.
    if split == datasets.Split.TRAIN:
        urls = _get_url_hashes(dl_paths["train_urls"])
    elif split == datasets.Split.VALIDATION:
        urls = _get_url_hashes(dl_paths["val_urls"])
    elif split == datasets.Split.TEST:
        urls = _get_url_hashes(dl_paths["test_urls"])
    else:
        logger.fatal("Unsupported split: %s", split)
    cnn = _find_files(dl_paths, "cnn", urls)
    dm = _find_files(dl_paths, "dm", urls)
    return cnn + dm


DM_SINGLE_CLOSE_QUOTE = "\u2019"  # unicode
DM_DOUBLE_CLOSE_QUOTE = "\u201d"
# acceptable ways to end a sentence
END_TOKENS = [
    ".",
    "!",
    "?",
    "...",
    "'",
    "`",
    '"',
    DM_SINGLE_CLOSE_QUOTE,
    DM_DOUBLE_CLOSE_QUOTE,
    ")",
]


def _read_text_file(text_file):
    lines = []
    with open(text_file, "r", encoding="utf-8") as f:
        for line in f:
            lines.append(line.strip())
    return lines


def _get_art_abs(story_file, tfds_version):
    """Get abstract (highlights) and article from a story file path."""
    # Based on https://github.com/abisee/cnn-dailymail/blob/master/
    #     make_datafiles.py

    lines = _read_text_file(story_file)

    # The github code lowercase the text and we removed it in 3.0.0.

    # Put periods on the ends of lines that are missing them
    # (this is a problem in the dataset because many image captions don't end in
    # periods; consequently they end up in the body of the article as run-on
    # sentences)
    def fix_missing_period(line):
        """Adds a period to a line that is missing a period."""
        if "@highlight" in line:
            return line
        if not line:
            return line
        if line[-1] in END_TOKENS:
            return line
        return line + " ."

    lines = [fix_missing_period(line) for line in lines]

    # Separate out article and abstract sentences
    article_lines = []
    highlights = []
    next_is_highlight = False
    for line in lines:
        if not line:
            continue  # empty line
        elif line.startswith("@highlight"):
            next_is_highlight = True
        elif next_is_highlight:
            highlights.append(line)
        else:
            article_lines.append(line)

    # Make article into a single string
    article = " ".join(article_lines)

    if tfds_version >= "2.0.0":
        abstract = "\n".join(highlights)
    else:
        abstract = " ".join(highlights)

    return article, abstract


class CnnDailymail(datasets.GeneratorBasedBuilder):
    """CNN/DailyMail non-anonymized summarization dataset."""

    BUILDER_CONFIGS = [
        CnnDailymailConfig(name=str(version), description="Plain text", version=version)
        for version in _SUPPORTED_VERSIONS
    ]

    def _info(self):
        # Should return a datasets.DatasetInfo object
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    _ARTICLE: datasets.Value("string"),
                    _HIGHLIGHTS: datasets.Value("string"),
                    "id": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage="https://github.com/abisee/cnn-dailymail",
            citation=_CITATION,
        )

    def _vocab_text_gen(self, paths):
        for _, ex in self._generate_examples(paths):
            yield " ".join([ex[_ARTICLE], ex[_HIGHLIGHTS]])

    def _split_generators(self, dl_manager):
        dl_paths = dl_manager.download_and_extract(_DL_URLS)
        train_files = _subset_filenames(dl_paths, datasets.Split.TRAIN)
        # Generate shared vocabulary

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kw={"files": train_files}),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kw={"files": _subset_filenames(dl_paths, datasets.Split.VALIDATION)},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kw={"files": _subset_filenames(dl_paths, datasets.Split.TEST)},
            ),
        ]

    def _generate_examples(self, files):
        for p in files:
            article, highlights = _get_art_abs(p, self.config.version)
            if not article or not highlights:
                continue
            fname = os.path.basename(p)
            yield fname, {
                _ARTICLE: article,
                _HIGHLIGHTS: highlights,
                "id": _get_hash_from_path(fname),
            }
