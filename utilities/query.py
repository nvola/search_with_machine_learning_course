# A simple client for querying driven by user input on the command line.  Has hooks for the various
# weeks (e.g. query understanding).  See the main section at the bottom of the file
from opensearchpy import OpenSearch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import argparse
import json
import os
from getpass import getpass
from urllib.parse import urljoin
import pandas as pd
import re
import fileinput
import logging
import fasttext
from pathlib import Path
from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(format='%(levelname)s:%(message)s')

MODEL_FPATH = Path(Path.cwd(), "week3", "models", "query_classifier.bin")
QC_MODEL = fasttext.load_model(str(MODEL_FPATH))

def normalize_query(query):
    # remove nonalphanum characters except space and underscore
    # and lowercase
    norm_query = " ".join([x.lower() for x in query.split(" ") if x.isalnum()])
    # trim excess space
    norm_query = re.sub(" +", " ", norm_query)
    return norm_query


def create_vector_query(model, query_string, n_results, filters=None):
    embedding = model.encode([query_string])

    knn_query = {
        "size": n_results,
        "query": {
            "knn": {
                "name_embedding": {
                    "vector": embedding[0].tolist(),
                    "k": n_results,
                }
            }
        }
    }

    if filters is not None:
        knn_query["post_filter"] = filters[0]

    return knn_query

# expects clicks and impressions to be in the row
def create_prior_queries_from_group(
        click_group):  # total impressions isn't currently used, but it mayb worthwhile at some point
    click_prior_query = ""
    # Create a string that looks like:  "query": "1065813^100 OR 8371111^89", where the left side is the doc id and the right side is the weight.  In our case, the number of clicks a document received in the training set
    if click_group is not None:
        for item in click_group.itertuples():
            try:
                click_prior_query += "%s^%.3f  " % (item.doc_id, item.clicks / item.num_impressions)

            except KeyError as ke:
                pass  # nothing to do in this case, it just means we can't find priors for this doc
    return click_prior_query


# expects clicks from the raw click logs, so value_counts() are being passed in
def create_prior_queries(doc_ids, doc_id_weights,
                         query_times_seen):  # total impressions isn't currently used, but it mayb worthwhile at some point
    click_prior_query = ""
    # Create a string that looks like:  "query": "1065813^100 OR 8371111^89", where the left side is the doc id and the right side is the weight.  In our case, the number of clicks a document received in the training set
    click_prior_map = ""  # looks like: '1065813':100, '8371111':809
    if doc_ids is not None and doc_id_weights is not None:
        for idx, doc in enumerate(doc_ids):
            try:
                wgt = doc_id_weights[doc]  # This should be the number of clicks or whatever
                click_prior_query += "%s^%.3f  " % (doc, wgt / query_times_seen)
            except KeyError as ke:
                pass  # nothing to do in this case, it just means we can't find priors for this doc
    return click_prior_query


# Hardcoded query here.  Better to use search templates or other query config.
def create_query(user_query, click_prior_query, filters, term_boosts=[], sort="_score", sortDir="desc", size=10, source=None, use_synonyms=False):
    match_field = "name" if not use_synonyms else "name.synonyms"
    query_obj = {
        'size': size,
        "sort": [
            {sort: {"order": sortDir}}
        ],
        "query": {
            "function_score": {
                "query": {
                    "bool": {
                        "must": [

                        ],
                        "should": [  #
                            {
                                "match": {
                                    match_field: {
                                        "query": user_query,
                                        "fuzziness": "1",
                                        "prefix_length": 2,
                                        # short words are often acronyms or usually not misspelled, so don't edit
                                        "boost": 0.01
                                    }
                                }
                            },
                            {
                                "match_phrase": {  # near exact phrase match
                                    "name.hyphens": {
                                        "query": user_query,
                                        "slop": 1,
                                        "boost": 50
                                    }
                                }
                            },
                            {
                                "multi_match": {
                                    "query": user_query,
                                    "type": "phrase",
                                    "slop": "6",
                                    "minimum_should_match": "2<75%",
                                    "fields": [f"{match_field}^10", "name.hyphens^10", "shortDescription^5",
                                               "longDescription^5", "department^0.5", "sku", "manufacturer", "features",
                                               "categoryPath"]
                                }
                            },
                            {
                                "terms": {
                                    # Lots of SKUs in the query logs, boost by it, split on whitespace so we get a list
                                    "sku": user_query.split(),
                                    "boost": 50.0
                                }
                            },
                            {  # lots of products have hyphens in them or other weird casing things like iPad
                                "match": {
                                    "name.hyphens": {
                                        "query": user_query,
                                        "operator": "OR",
                                        "minimum_should_match": "2<75%"
                                    }
                                }
                            },
                        ],
                        "minimum_should_match": 1,
                        "filter": filters  #
                    }
                },
                "boost_mode": "multiply",  # how _score and functions are combined
                "score_mode": "sum",  # how functions are combined
                "functions": [
                    {
                        "filter": {
                            "exists": {
                                "field": "salesRankShortTerm"
                            }
                        },
                        "gauss": {
                            "salesRankShortTerm": {
                                "origin": "1.0",
                                "scale": "100"
                            }
                        }
                    },
                    {
                        "filter": {
                            "exists": {
                                "field": "salesRankMediumTerm"
                            }
                        },
                        "gauss": {
                            "salesRankMediumTerm": {
                                "origin": "1.0",
                                "scale": "1000"
                            }
                        }
                    },
                    {
                        "filter": {
                            "exists": {
                                "field": "salesRankLongTerm"
                            }
                        },
                        "gauss": {
                            "salesRankLongTerm": {
                                "origin": "1.0",
                                "scale": "1000"
                            }
                        }
                    },
                    {
                        "script_score": {
                            "script": "0.0001"
                        }
                    }
                ]

            }
        }
    }

    if term_boosts is not None:
        for term_boost in term_boosts:
            query_obj["query"]["function_score"]["query"]["bool"]["should"].append(
                {
                    "terms": {
                        term_boost["field"]: term_boost["values"],
                        "boost": term_boost["boost"],
                    }
                }
            )

    if click_prior_query is not None and click_prior_query != "":
        query_obj["query"]["function_score"]["query"]["bool"]["should"].append({
            "query_string": {
                # This may feel like cheating, but it's really not, esp. in ecommerce where you have all this prior data,  You just can't let the test clicks leak in, which is why we split on date
                "query": click_prior_query,
                "fields": ["_id"]
            }
        })
    if user_query == "*" or user_query == "#":
        # replace the bool
        try:
            query_obj["query"] = {"match_all": {}}
        except:
            print("Couldn't replace query for *")
    if source is not None:  # otherwise use the default and retrieve all source
        query_obj["_source"] = source
    return query_obj


def search(client, user_query, index="bbuy_products", model=None, sort="_score", sortDir="desc", use_synonyms=False, use_filters=False, use_boosts=False, use_vectors=False, k=3):
    #### W3: classify the query
    filter_labels = []
    if use_filters:
        query_pred = QC_MODEL.predict(normalize_query(user_query), k=3)
        for l, p in zip(query_pred[0], query_pred[1]):
            if p > 0.3:
                filter_labels.append(l[9:])
    #### W3: create filters and boosts
    if (use_filters) & (len(filter_labels) > 0):
        filters= [{
            "terms": {
                "categoryPathIds": filter_labels
            }
        }
        ]
    else:
        filters = None

    if (use_boosts) & (len(filter_labels) > 0):
        term_boosts = [
            {
                "field": "categoryPathIds",
                "values": filter_labels,
                "boost": 0.05,
            },
            {
                "field": "categoryLeaf",
                "values":[filter_labels[0]],
                "boost": 0.025,
            }
        ]
    else:
        term_boosts = None

    # Note: you may also want to modify the `create_query` method above
    if use_vectors:
        query_obj = create_vector_query(model, user_query, k, filters)
    else:
        query_obj = create_query(user_query, click_prior_query=None, filters=filters, term_boosts=term_boosts, sort=sort, sortDir=sortDir, source=["name", "shortDescription"], use_synonyms=use_synonyms)
    
    # print(query_obj)
    # logger.info(query_obj)
    def check_valid_response(response):
        return response and response['hits']['hits'] and len(response['hits']['hits']) > 0

    response = client.search(query_obj, index=index)
    valid_response = check_valid_response(response)
    if valid_response:
        hits = response['hits']['hits']
        # print(json.dumps(response, indent=2))
        total_hits = response['hits']['total']['value']
        print(f"total hits: {total_hits}, max score: {response['hits']['max_score']}\n")
        print(f"returned hits {len(hits)}")

    if (valid_response) & (use_filters) & (use_vectors) & (len(hits) < k) & (total_hits > k):
        print("not enough hits generated, trying again")
        # retry query if filtering returned fewer results than requested
        # but more results exist
        enough_results = False
        new_k = k 

        while not enough_results:
            new_k = k * 2
            query_obj = create_vector_query(model, user_query, new_k, filters)
            response = opensearch.search(query_obj, index=index_name)
            if check_valid_response(response):
                new_hits = response['hits']['total']['value']
                if (new_hits == k) or (new_hits == total_hits):
                    enough_results = True

    if valid_response:
        fields = ['productId', 'sku', 'name', 'type', 'shortDescription', 'department', 'categoryPathIds']
        for hit in hits:
            print({field: hit["_source"][field] for field in fields})
            print("\n")


if __name__ == "__main__":
    host = 'localhost'
    port = 9200
    auth = ('admin', 'admin')  # For testing only. Don't store credentials in code.
    parser = argparse.ArgumentParser(description='Build LTR.')
    general = parser.add_argument_group("general")
    general.add_argument("-i", '--index', default="bbuy_products",
                         help='The name of the main index to search')
    general.add_argument("-s", '--host', default="localhost",
                         help='The OpenSearch host name')
    general.add_argument("-p", '--port', type=int, default=9200,
                         help='The OpenSearch port')
    general.add_argument('--user',
                         help='The OpenSearch admin.  If this is set, the program will prompt for password too. If not set, use default of admin/admin')
    general.add_argument(
        '--use_synonyms',
        help='If true, query name.synonyms field instead of name',
        action="store_true",
        default=False,
    )
    general.add_argument(
        '--use_filters',
        help='If true, use categoryPathIds filters',
        action="store_true",
        default=False,
    )
    general.add_argument(
        '--use_boosts',
        help='If true, use categoryPathIds boosts',
        action="store_true",
        default=False,
    )
    general.add_argument(
        '--vector',
        help='If true, use vector query',
        action="store_true",
        default=False,
    )
    general.add_argument(
        '--k',
        help='Number of nearest neighbors to pull for knn',
        default=3,
        type=int,
    )

    args = parser.parse_args()

    if len(vars(args)) == 0:
        parser.print_usage()
        exit()

    host = args.host
    port = args.port
    if args.user:
        password = getpass()
        auth = (args.user, password)

    base_url = "https://{}:{}/".format(host, port)
    opensearch = OpenSearch(
        hosts=[{'host': host, 'port': port}],
        http_compress=True,  # enables gzip compression for request bodies
        http_auth=auth,
        # client_cert = client_cert_path,
        # client_key = client_key_path,
        use_ssl=True,
        verify_certs=False,  # set to true if you have certs
        ssl_assert_hostname=False,
        ssl_show_warn=False,

    )
    index_name = args.index
    query_prompt = "\nEnter your query (type 'Exit' to exit or hit ctrl-c):"
    print(query_prompt)
    import sys 

    if args.vector:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    else:
        model = None

    for line in sys.stdin:
        query = line.rstrip()
        if query == "Exit":
            break
        search(
            client=opensearch,
            user_query=query,
            index=index_name,
            model=model,
            use_synonyms=args.use_synonyms,
            use_filters=args.use_filters,
            use_boosts=args.use_boosts,
            use_vectors=args.vector,
            k=args.k
        )

        print(query_prompt)

    