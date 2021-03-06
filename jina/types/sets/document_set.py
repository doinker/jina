from collections.abc import MutableSequence
from typing import Union, Sequence, Iterable, Tuple

import numpy as np
from google.protobuf.pyext._message import RepeatedCompositeContainer

from ...proto.jina_pb2 import DocumentProto

if False:
    from ..document import Document

__all__ = ['DocumentSet']


class DocumentSet(MutableSequence):
    """:class:`DocumentSet` is a mutable sequence of :class:`Document`,
    it gives an efficient view of a list of Document. One can iterate over it like
    a generator but ALSO modify it, count it, get item.
    """

    def __init__(self, docs_proto: Union['RepeatedCompositeContainer', Sequence['Document']]):
        super().__init__()
        self._docs_proto = docs_proto
        self._docs_map = {}

    def insert(self, index: int, doc: 'Document') -> None:
        self._docs_proto.insert(index, doc.as_pb_object)

    def __setitem__(self, key, value: 'Document'):
        if isinstance(key, int):
            self._docs_proto[key].CopyFrom(value)
        elif isinstance(key, str):
            self._docs_map[key].CopyFrom(value)
        else:
            raise IndexError(f'do not support this index {key}')

    def __delitem__(self, index):
        del self._docs_proto[index]

    def __len__(self):
        return len(self._docs_proto)

    def __iter__(self):
        from ..document import Document
        for d in self._docs_proto:
            yield Document(d)

    def __getitem__(self, item):
        from ..document import Document
        if isinstance(item, int):
            return Document(self._docs_proto[item])
        elif isinstance(item, str):
            return Document(self._docs_map[item])
        else:
            raise IndexError(f'do not support this index {item}')

    def append(self, doc: 'Document'):
        self._docs_proto.append(doc.as_pb_object)

    def add(self, doc: 'Document'):
        """Shortcut to :meth:`append`, do not override this method """
        self.append(doc)

    def extend(self, iterable: Iterable['Document']) -> None:
        self._docs_proto.extend(doc.as_pb_object for doc in iterable)

    def clear(self):
        del self._docs_proto[:]

    def reverse(self):
        """In-place reverse the sequence """
        if isinstance(self._docs_proto, RepeatedCompositeContainer):
            size = len(self._docs_proto)
            hi_idx = size - 1
            for i in range(int(size / 2)):
                tmp = DocumentProto()
                tmp.CopyFrom(self._docs_proto[hi_idx])
                self._docs_proto[hi_idx].CopyFrom(self._docs_proto[i])
                self._docs_proto[i].CopyFrom(tmp)
                hi_idx -= 1
        elif isinstance(self._docs_proto, list):
            self._docs_proto.reverse()

    def build(self):
        """Build a doc_id to doc mapping so one can later index a Document using
        doc_id as string key
        """
        self._docs_map = {d.id: d for d in self._docs_proto}

    def sort(self, *args, **kwargs):
        self._docs_proto.sort(*args, **kwargs)

    @property
    def all_embeddings(self) -> Tuple['np.ndarray', 'DocumentSet', 'DocumentSet']:
        """Return all embeddings from every document in this set as a ndarray

        :return a tuple of embedding in :class:`np.ndarray`,
                the corresponding documents in a :class:`DocumentSet`,
                and the documents have no embedding in a :class:`DocumentSet`.
        """
        return self._extract_docs('embedding')

    @property
    def all_contents(self) -> Tuple['np.ndarray', 'DocumentSet', 'DocumentSet']:
        """Return all embeddings from every document in this set as a ndarray

        :return a tuple of embedding in :class:`np.ndarray`,
                the corresponding documents in a :class:`DocumentSet`,
                and the documents have no contents in a :class:`DocumentSet`.
        """
        return self._extract_docs('content')

    def _extract_docs(self, attr: str) -> Tuple['np.ndarray', 'DocumentSet', 'DocumentSet']:
        contents = []
        docs_pts = []
        bad_docs = []

        for doc in self:
            content = getattr(doc, attr)

            if content is not None:
                contents.append(content)
                docs_pts.append(doc)
            else:
                bad_docs.append(doc)

        contents = np.stack(contents) if contents else None
        return contents, DocumentSet(docs_pts), DocumentSet(bad_docs)

    def __bool__(self):
        """To simulate ```l = []; if l: ...``` """
        return bool(len(self))
 

class MultiModalDocumentSet(DocumentSet):
    """:class:`MultiModalDocumentSet` is a mutable sequence of :class:`Document`,
    It wraps itself a DocumentSet to guarantee that it iterates guaranteeing that the generated
    documents fulfill the MultiModal Document specifications
    """

    def __init__(self, document_set:  Union[DocumentSet, 'RepeatedCompositeContainer', Sequence['Document']]):
        if isinstance(document_set, DocumentSet):
            super().__init__(docs_proto=document_set._docs_proto)
        else:
            super().__init__(docs_proto=document_set)

    def __iter__(self):
        from ..document.multimodal import MultimodalDocument
        for d in self._docs_proto:
            yield MultimodalDocument(d)
