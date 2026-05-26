#!/usr/bin/env nextflow
nextflow.enable.dsl=2

process DOWNLOAD {
    container 'python:3.11'

    output:
    path 'trials_raw.csv', emit: raw_csv

    script:
    """
    pip install -q -r ${projectDir}/requirements.txt
    mkdir -p data
    cp ${projectDir}/data/studies.txt data/studies.txt
    cp ${projectDir}/data/brief_summaries.txt data/brief_summaries.txt
    python ${projectDir}/bin/download.py
    cp data/trials_raw.csv .
    """
}

process PREPROCESS {
    container 'python:3.11'

    input:
    path raw_csv

    output:
    path 'trials_clean.csv', emit: clean_csv

    script:
    """
    pip install -q -r ${projectDir}/requirements.txt
    mkdir -p data
    cp ${raw_csv} data/trials_raw.csv
    python ${projectDir}/bin/preprocess.py
    cp data/trials_clean.csv .
    """
}

process EMBED {
    container 'python:3.11'

    input:
    path clean_csv

    output:
    path 'tfidf_matrix.npz',  emit: matrix
    path 'feature_names.txt', emit: feature_names

    script:
    """
    pip install -q -r ${projectDir}/requirements.txt
    mkdir -p data
    cp ${clean_csv} data/trials_clean.csv
    python ${projectDir}/bin/embed.py
    cp data/tfidf_matrix.npz .
    cp data/feature_names.txt .
    """
}

process CLUSTER {
    container 'python:3.11'

    input:
    path matrix
    path feature_names
    path clean_csv

    output:
    path 'cluster_labels.csv',   emit: labels
    path 'cluster_keywords.json', emit: keywords

    script:
    """
    pip install -q -r ${projectDir}/requirements.txt
    mkdir -p data
    cp ${matrix}        data/tfidf_matrix.npz
    cp ${feature_names} data/feature_names.txt
    cp ${clean_csv}     data/trials_clean.csv
    python ${projectDir}/bin/cluster.py
    cp data/cluster_labels.csv   .
    cp data/cluster_keywords.json .
    """
}

process VISUALIZE {
    container 'python:3.11'
    publishDir "${projectDir}/output", mode: 'copy'

    input:
    path matrix
    path labels
    path keywords
    path clean_csv

    output:
    path 'clusters.html'

    script:
    """
    pip install -q -r ${projectDir}/requirements.txt
    mkdir -p data output
    cp ${matrix}    data/tfidf_matrix.npz
    cp ${labels}    data/cluster_labels.csv
    cp ${keywords}  data/cluster_keywords.json
    cp ${clean_csv} data/trials_clean.csv
    python ${projectDir}/bin/visualize.py
    cp output/clusters.html .
    """
}

workflow {
    DOWNLOAD()
    PREPROCESS(DOWNLOAD.out.raw_csv)

    // clean_csv feeds EMBED, CLUSTER, and VISUALIZE — tap forks the channel
    PREPROCESS.out.clean_csv
        .tap { clean_for_embed }
        .tap { clean_for_cluster }
        .set { clean_for_viz }

    EMBED(clean_for_embed)

    // matrix feeds both CLUSTER and VISUALIZE — tap forks the channel
    EMBED.out.matrix
        .tap { matrix_for_cluster }
        .set { matrix_for_viz }

    CLUSTER(matrix_for_cluster, EMBED.out.feature_names, clean_for_cluster)

    VISUALIZE(
        matrix_for_viz,
        CLUSTER.out.labels,
        CLUSTER.out.keywords,
        clean_for_viz
    )
}
