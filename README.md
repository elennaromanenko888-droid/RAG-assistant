# Case 2: RAG-ассистент на ChromaDB

Учебный проект для задания №2: скрипт загружает текст, делит его на чанки, строит эмбеддинги, сохраняет их в ChromaDB и ищет релевантные фрагменты по вопросу.

## Что внутри

- `rag_assistant.py` - основной Python-скрипт.
- `data/knowledge_base.txt` - пример текстовой базы знаний.
- `requirements.txt` - зависимости проекта.
- `chroma_db/` - локальная база ChromaDB, создается после запуска и не добавляется в Git.

## Как запустить

Перейдите в папку проекта:

```powershell
cd case-2-rag-assistant
```

Создайте виртуальное окружение:

```powershell
python -m venv .venv
```

Активируйте его:

```powershell
.\.venv\Scripts\Activate.ps1
```

Установите ChromaDB:

```powershell
python -m pip install -r requirements.txt
```

Запустите ассистента с вопросом по умолчанию:

```powershell
python rag_assistant.py
```

Или задайте свой вопрос:

```powershell
python rag_assistant.py --question "Как ChromaDB помогает в RAG?"
```

## Как это работает

1. **Загрузка текста**: скрипт читает файл `data/knowledge_base.txt`.
2. **Чанкинг**: большой текст делится на небольшие фрагменты по словам.
3. **Эмбеддинги**: каждый чанк превращается в числовой вектор.
4. **ChromaDB**: чанки и их векторы сохраняются в локальную векторную базу.
5. **Поиск**: вопрос тоже превращается в вектор, после чего ChromaDB возвращает ближайшие чанки.

В этом проекте используется простой локальный hash-based embedding. Он нужен для обучения и демонстрации пайплайна без API-ключей. В реальных проектах вместо него обычно подключают embedding-модель, например `sentence-transformers` или API внешнего провайдера.

## Пример результата

После запуска в консоли появятся:

- путь к текстовой базе;
- количество созданных чанков;
- вопрос;
- короткий ответ на основе лучшего найденного чанка;
- список найденных фрагментов с расстоянием похожести.

## Что приложить к сдаче

1. Загрузите папку проекта на GitHub.
2. В ответе на платформе сдачи укажите ссылку на GitHub-репозиторий.
3. Сделайте скриншот консоли после команды:

```powershell
python rag_assistant.py --question "Что такое чанкинг и зачем он нужен?"
```

4. Приложите этот скриншот к сдаче.

## Как загрузить на GitHub

Создайте пустой репозиторий на GitHub с названием `case-2-rag-assistant`, затем выполните команды в папке проекта:

```powershell
git config user.name "Ваше Имя"
git config user.email "your-email@example.com"
git add .
git commit -m "Add case 2 RAG assistant"
git branch -M main
git remote add origin https://github.com/USERNAME/case-2-rag-assistant.git
git push -u origin main
```

Замените `USERNAME`, имя и email на свои данные.
