
# Analisador de Saques no Tênis

<img src="./video.gif"/>
Este é o código fonte do projeto apresentado neste vídeo:
https://www.instagram.com/reel/C8Ndmh2OAze/

Este é um projeto pessoal que desenvolvi para me ajudar em minhas sessões de treino de Saque de tênis. Foi construído 100% em Python, utilizando as bibliotecas Streamlit, OpenCV, MediaPipe e, é claro, pode ser adaptado para ser aplicado em qualquer modalidade esportiva: futebol, basquete, corrida, skate, musculação...  
  
O tutorial completo de como executá-lo e adaptá-lo esta disponível na [Trilha Visão Computacional com Python](https://hub.asimov.academy/projeto/analisador-de-saques-no-tenis-com-visao-computacional/).


## Como executar?

1. Instale as dependências com `pip install -r requirements.txt`
2. Execute `streamlit run home.py`
3. Você pode utilizar os scripts adicionais `serve_cut.py` e `normalize_lands.py` para analisar seus próprios vídeos. Os cortes são salvos na pasta `serve`