import os
import numpy as np

from jina import Flow, Document

callback_was_called = False


def get_index_flow():
    num_shards = 2
    f = Flow() \
        .add(
        uses='vectorindexer.yml',
        shards=num_shards,
        separated_workspace=True,
    )
    return f


def get_flow():
    num_shards = 2
    f = Flow() \
        .add(
        uses='vectorindexer.yml',
        shards=num_shards,
        separated_workspace=True,
        uses_after='merge_all_exec.yml',
        polling='all',
        timeout_ready='-1'
    )
    return f


def test_sharding_empty_index(tmpdir, execution_number):
    print(f'WORKSPACE = {tmpdir}')
    os.environ['JINA_TEST_1229_WORKSPACE'] = os.path.abspath(tmpdir)

    f = get_index_flow()

    num_docs = 1
    data = []
    for i in range(num_docs):
        with Document() as doc:
            doc.content = f'data {i}'
            doc.embedding = np.array([i])
            data.append(doc)

    with f:
        f.index(data)

    f.close()

    f = get_flow()

    num_query = 10
    query = []
    for i in range(num_query):
        with Document() as doc:
            doc.content = f'query {i}'
            doc.embedding = np.array([i])
            query.append(doc)

    def callback(result):
        global callback_was_called
        callback_was_called = True
        assert len(result.docs) == num_query
        for d in result.docs:
            print(f'document: {d.content} with {len(d.matches)} matches')
            assert len(list(d.matches)) == num_docs
            print(f'matches = {[f"content: {m.content}, score: {m.score.value}" for m in d.matches]}')

    with f:
        f.search(query, output_fn=callback)
    assert callback_was_called
