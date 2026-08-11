FROM elasticsearch:8.15.5

RUN bin/elasticsearch-plugin install --batch \
    https://get.infini.cloud/elasticsearch/analysis-ik/8.15.5

COPY analysis-ik/ecommerce.dic config/analysis-ik/ecommerce.dic
COPY analysis-ik/IKAnalyzer.cfg.xml config/analysis-ik/IKAnalyzer.cfg.xml
