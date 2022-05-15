# coding=utf-8
# Copyright 2022 The HuggingFace Datasets Authors and the current dataset script contributor.
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
"""ner."""

import os

import datasets
import pandas as pd

_CITATION = """None"""

_DESCRIPTION = """ner"""

_HOMEPAGE = "https://github.com/JunnYu/datasets_jy"

_LICENSE = "MIT"

_EN_BASE_URL = "en/"
_ZH_BASE_URL = "zh/"

_URLs = {
    # en
    "ace2004": _EN_BASE_URL + "ace2004.zip",
    "ace2005": _EN_BASE_URL + "ace2005.zip",
    "conll03": _EN_BASE_URL + "conll03.zip",
    "cadec": _EN_BASE_URL + "cadec.zip",
    "genia": _EN_BASE_URL + "genia.zip",
    # zh
    "china_people_daily": _ZH_BASE_URL + "china_people_daily.zip",
    "cluener": _ZH_BASE_URL + "cluener.zip",
    "medical": _ZH_BASE_URL + "medical.zip",
    "msra": _ZH_BASE_URL + "msra.zip",
    "onto4": _ZH_BASE_URL + "onto4.zip",
    "cmeee": _ZH_BASE_URL + "cmeee.zip",
    "cail2021": _ZH_BASE_URL + "cail2021.zip",
    "weibo": _ZH_BASE_URL + "weibo.zip",
    "cner": _ZH_BASE_URL + "cner.zip",
}

_LABEL_MAP = {
    # en
    "ace2004": ["FAC", "GPE", "LOC", "ORG", "PER", "VEH", "WEA"],
    "ace2005": ["FAC", "GPE", "LOC", "ORG", "PER", "VEH", "WEA"],
    "conll03": ["LOC", "MISC", "ORG", "PER"],
    "cadec": ["ADR"],
    "genia": ["cell_line", "cell_type", "DNA", "protein", "RNA"],
    # zh
    "china_people_daily": ["LOC", "ORG", "PER"],
    "cluener": [
        "ADDRESS",
        "BOOK",
        "COMPANY",
        "GAME",
        "GOVERNMENT",
        "MOVIE",
        "NAME",
        "ORGANIZATION",
        "POSITION",
        "SCENE",
    ],
    "medical":
    ["Drug", "Drug_Category", "Medical_Examination", "Operation", "Symptom"],
    "msra": ["NR", "NS", "NT"],
    "onto4": ["GPE", "LOC", "ORG", "PER"],
    "cmeee": ["bod", "dep", "dis", "dru", "equ", "ite", "mic", "pro", "sym"],
    "cail2021": [
        "NASI",
        "NATS",
        "NCGV",
        "NCSM",
        "NCSP",
        "NHCS",
        "NHVI",
        "NO",
        "NS",
        "NT",
    ],
    "weibo": [
        "GPE.NAM",
        "GPE.NOM",
        "LOC.NAM",
        "LOC.NOM",
        "ORG.NAM",
        "ORG.NOM",
        "PER.NAM",
        "PER.NOM",
    ],
    "cner": ["CONT", "EDU", "LOC", "NAME", "ORG", "PRO", "RACE", "TITLE"],
}


class NER(datasets.GeneratorBasedBuilder):
    """NER"""

    VERSION = datasets.Version("1.1.0")

    BUILDER_CONFIGS = [
        # en
        # nested
        datasets.BuilderConfig(
            name="en_ace2004",
            version=VERSION,
            description="ace2004: Nested NER"),
        datasets.BuilderConfig(
            name="en_ace2005",
            version=VERSION,
            description="ace2005: Nested NER"),
        datasets.BuilderConfig(
            name="en_genia", version=VERSION, description="genia: Nested NER"),
        # flat
        datasets.BuilderConfig(
            name="en_conll03",
            version=VERSION,
            description="conll03: Flat NER"),
        # discontinuous
        datasets.BuilderConfig(
            name="en_cadec",
            version=VERSION,
            description="cadec: Discontinuous NER"),
        # zh
        # nested
        datasets.BuilderConfig(
            name="zh_china_people_daily",
            version=VERSION,
            description="china_people_daily: Nested NER", ),
        datasets.BuilderConfig(
            name="zh_cmeee", version=VERSION, description="cmeee: Nested NER"),
        # flat
        datasets.BuilderConfig(
            name="zh_cluener",
            version=VERSION,
            description="cluener: Flat NER"),
        datasets.BuilderConfig(
            name="zh_medical",
            version=VERSION,
            description="medical: Flat NER"),
        datasets.BuilderConfig(
            name="zh_msra", version=VERSION, description="msra: Flat NER"),
        datasets.BuilderConfig(
            name="zh_onto4", version=VERSION, description="onto4: Flat NER"),
        datasets.BuilderConfig(
            name="zh_weibo", version=VERSION, description="weibo: Flat NER"),
        datasets.BuilderConfig(
            name="zh_cail2021",
            version=VERSION,
            description="cail2021: Nested NER"),
        datasets.BuilderConfig(
            name="zh_cner", version=VERSION, description="zh_cner: Flat NER"),
    ]

    DEFAULT_CONFIG_NAME = "zh_cail2021"

    def _info(self):
        label = _LABEL_MAP[self.config.name[3:]]

        if self.config.name == "en_cadec":
            self.discontinuous = True
            features = datasets.Features({
                "id": datasets.Value("string"),
                "tokens":
                datasets.features.Sequence(datasets.Value("string")),
                "text": datasets.Value("string"),
                "ner_span": datasets.Value("string"),
            })
        else:
            self.discontinuous = False
            features = datasets.Features({
                "id": datasets.Value("string"),
                "tokens":
                datasets.features.Sequence(datasets.Value("string")),
                "text": datasets.Value("string"),
                "ner_span": datasets.features.Sequence({
                    "type": datasets.ClassLabel(names=label),
                    "start": datasets.Value("int64"),
                    "end": datasets.Value("int64"),
                }),
            })
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            # Here we define them above because they are different between the two configurations
            features=features,
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION, )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        name = self.config.name[3:]
        my_urls = _URLs[name]

        data_dir = dl_manager.download_and_extract(my_urls)

        outputs = [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "train.tsv"),
                    "split": "train",
                }, ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "dev.tsv"),
                    "split": "dev",
                }, ),
        ]
        test_file = os.path.join(data_dir, "test.tsv")
        if os.path.exists(test_file):
            outputs.append(
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    # These kwargs will be passed to _generate_examples
                    gen_kwargs={
                        "filepath": test_file,
                        "split": "test",
                    }, ))
        return outputs

    def _generate_examples(
            # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
            self,
            filepath,
            split, ):
        """Yields examples as (key, example) tuples."""
        d = pd.read_csv(filepath, sep="\t", converters={"label": eval})

        for id_, data in d.iterrows():
            if self.discontinuous:
                ner_span = str(data["label"])
            else:
                ner_span = [{
                    "type": tp,
                    "start": s,
                    "end": e
                } for tp, s, e in data["label"]]

            yield id_, {
                "id": str(id_),
                "tokens": data["text"].split(),
                "text": data["text"].replace(" ", ""),
                "ner_span": ner_span,
            }
