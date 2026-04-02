# 📚 Post Graduation Utils - For Students

Bem-vindo! Esta é sua biblioteca Python com utilitários para Machine Learning, Deep Learning e Generative AI.

## 🚀 Quick Start (3 passos)

### 1. Instalar

```bash
pip install post-graduation-utils
```

### 2. Importar

```python
from post_graduation_utils import core, ml, deep_learning, genai
```

### 3. Usar!

```python
# Seu código aqui
```

---

## 📦 Instalação Personalizada

Escolha o que você precisa:

### Apenas Machine Learning
```bash
pip install "post-graduation-utils[ml]"
```
✅ sklearn, xgboost, lightgbm

### Apenas Deep Learning
```bash
pip install "post-graduation-utils[deep_learning]"
```
✅ PyTorch, TensorFlow, Keras

### Apenas Generative AI
```bash
pip install "post-graduation-utils[genai]"
```
✅ OpenAI, LangChain, HuggingFace

### Tudo (Recomendado!)
```bash
pip install "post-graduation-utils[all]"
```
✅ Todas as dependências acima

---

## 📁 O que cada módulo oferece?

### `core` - Utilitários Compartilhados
Funções gerais usadas em todo o projeto

```python
from post_graduation_utils import core
```

### `ml` - Machine Learning
Preprocessing, modelos, e avaliação

```python
from post_graduation_utils.ml import preprocessing, models
```

### `deep_learning` - Deep Learning
Arquiteturas, treinamento, transfer learning, visualização de redes

```python
from post_graduation_utils.deep_learning import draw_neural_network
```

### `genai` - Generative AI
LLMs, RAG, e prompt engineering

```python
from post_graduation_utils.genai import llm, rag
```

### 🏫 `puc` ou `ibmec` - Seu Módulo Institucional
Configurações específicas para sua instituição

```python
# Para alunos PUC
from post_graduation_utils.puc import config

# Para alunos IBMEC
from post_graduation_utils.ibmec import config
```

---

## 💡 Exemplos

### Verificar sua instituição

```python
from post_graduation_utils.puc import config as puc_config
print(puc_config.PUCConfig.get_info())
# Saída: PUC Utils v1.0.0
```

### Rodar exemplo básico

```bash
python examples/example_basic.py
```

---

## 📖 Documentação Completa

- **[Guia de Instalação](docs/INSTALLATION_GUIDE.md)** - Instruções detalhadas
- **[Começando](docs/GETTING_STARTED.md)** - Primeiros passos
- **[Arquitetura](docs/ARCHITECTURE.md)** - Como o projeto é organizado
- **[README Principal](README.md)** - Documentação técnica completa

---

## 🧪 Testes

Testar se tudo está funcionando:

```bash
pytest tests/
```

---

## 🐛 Problemas?

### Erro de importação?
```bash
pip install post-graduation-utils
```

### Faltam dependências?
```bash
pip install "post-graduation-utils[all]"
```

### Versão desatualizada?
```bash
pip install --upgrade post-graduation-utils
```

---

## 📝 Notas Importantes

- ✅ A biblioteca é modular - instale apenas o que precisa
- ✅ Use `pip install -e .` se estiver desenvolvendo
- ✅ Consulte os exemplos na pasta `examples/`
- ✅ Leia a documentação em `docs/`

---

## 🎓 Para Instrutores

Documentação técnica completa está em:
- [README.md](README.md) - Documentação técnica
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - Arquitetura do projeto
- [docs/INSTALLATION_GUIDE.md](docs/INSTALLATION_GUIDE.md) - Guia de instalação

---

## ❓ Dúvidas?

1. Verifique a documentação em `docs/`
2. Olhe os exemplos em `examples/`
3. Pergunte em aula ou office hours!

---

**Bom aprendizado!** 🚀
