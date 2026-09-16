import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent
# Промпты один в один с продуктового сервера: системный и шаблон сообщения пользователя.
SYSTEM_PROMPT_PATH = BASE / "Promt" / "System promt.md"
USER_PROMPT_PATH = BASE / "Promt" / "User promt.md"
# На сервере [Date] и [Users] подставляет n8n из базы, на стенде их нет.
UNKNOWN_FIELD = "Уточнити"
MEETING_DIR = BASE / "Meeting"
RESULTS_DIR = BASE / "results"
DEFAULT_TRANSCRIPT = MEETING_DIR / "Зустріч МТА.txt"

ENV_FILES = [
    BASE / ".env",
    Path(r"D:\Documents\Google Drive\AI Project\Study\Openroute\.env"),
]
KEY_NAMES = ("OPENROUTER_API_KEY", "OPEN_ROUTER_API")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
TEMPERATURE = 0
REQUEST_TIMEOUT = 1800
OUTPUT_RESERVE_TOKENS = 4000

# Список кандидатов. kind: openrouter или ollama. enabled: участвует ли в прогоне.
MODELS = [
    {"label": "gpt-oss-20b", "kind": "openrouter", "id": "openai/gpt-oss-20b", "enabled": True},
    {"label": "gpt-oss-120b", "kind": "openrouter", "id": "openai/gpt-oss-120b", "enabled": True},
    {"label": "qwen3.5-27b", "kind": "openrouter", "id": "qwen/qwen3.5-27b", "enabled": True},
    {"label": "gpt-4.1-mini", "kind": "openrouter", "id": "openai/gpt-4.1-mini", "enabled": True},
    {"label": "gpt-5.6-luna", "kind": "openrouter", "id": "openai/gpt-5.6-luna", "enabled": True},
    {"label": "qwen3.8-27b", "kind": "openrouter", "id": "qwen/qwen3.8-27b", "enabled": True},
    {"label": "gpt-oss-20b-local", "kind": "ollama", "id": "gpt-oss-20b-GGUF:Q4_K_M", "enabled": False},
    {"label": "qwen3.5-local", "kind": "ollama", "id": "qwen3.5:latest", "enabled": False},
]

BLIND_LABELS = ["А", "Б", "В", "Г", "Д", "Е", "Ж", "З", "И", "К"]

SECTION_TITLES = {
    1: "Загальна інформація",
    2: "Мета зустрічі",
    3: "Ключові питання",
    4: "Технічні та специфічні деталі",
    5: "Прийняті рішення",
    6: "План дій",
    7: "Наступні кроки",
    8: "Примітки",
}

RUBRIC = """## Что нужно сделать

Ниже стенограмма одной встречи и несколько протоколов, сделанных по ней разными
моделями. Какой моделью сделан каждый протокол, не указано намеренно.

Оцени каждый протокол по четырём критериям, по шкале от 1 до 5, и обоснуй оценку
ссылками на конкретные места в стенограмме:

1. **Точность фактов.** Всё ли, что написано в протоколе, действительно есть в
   стенограмме. Цифры, даты, имена, суммы, решения.
2. **Полнота.** Не потеряны ли существенные темы, решения и договорённости.
   Отдельно отметь, не обрывается ли протокол на первой части встречи.
3. **Отсутствие выдумок.** Есть ли утверждения, которых в стенограмме нет.
   По инструкции модель обязана писать «Уточнити» там, где факта нет, а не
   додумывать. Выдуманный факт это грубая ошибка, а не мелкая.
4. **Пригодность без правки.** Можно ли отдать такой протокол участникам встречи
   как есть, без редактирования.

В конце дай сводную таблицу с оценками и одну фразу: какой протокол лучший и
почему именно он.
"""


def read_text(path: Path) -> str:
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def load_api_key() -> str | None:
    for env_file in ENV_FILES:
        if not env_file.exists():
            continue
        for line in read_text(env_file).splitlines():
            name, _, value = line.partition("=")
            if name.strip() in KEY_NAMES:
                key = value.strip().strip('"').strip("'")
                if key:
                    return key
    for name in KEY_NAMES:
        key = os.environ.get(name)
        if key:
            return key.strip()
    return None


def estimate_tokens(text: str) -> int:
    return int(len(text) / 2.6)


def post_json(url: str, payload: dict, headers: dict, timeout: int) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def ollama_context_limit(model_id: str) -> int | None:
    try:
        info = post_json(f"{OLLAMA_HOST}/api/show", {"model": model_id}, {}, 30)
    except Exception:
        return None
    for key, value in (info.get("model_info") or {}).items():
        if key.endswith(".context_length") and isinstance(value, int):
            return value
    return None


def call_ollama(model: dict, system: str, user: str) -> dict:
    needed = estimate_tokens(system + user) + OUTPUT_RESERVE_TOKENS
    limit = ollama_context_limit(model["id"])
    num_ctx = needed if limit is None else min(needed, limit)
    warning = None
    if limit is not None and needed > limit:
        warning = f"вход ~{needed} токенов, модель держит {limit}: часть стенограммы не поместится"

    payload = {
        "model": model["id"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": TEMPERATURE, "num_ctx": num_ctx},
    }
    data = post_json(f"{OLLAMA_HOST}/api/chat", payload, {}, REQUEST_TIMEOUT)
    prompt_tokens = data.get("prompt_eval_count")
    truncated = bool(prompt_tokens and prompt_tokens >= num_ctx - 50)
    return {
        "text": (data.get("message") or {}).get("content", ""),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": data.get("eval_count"),
        "cost": 0.0,
        "num_ctx": num_ctx,
        "context_limit": limit,
        "truncated": truncated,
        "warning": warning,
    }


def call_openrouter(model: dict, system: str, user: str, api_key: str) -> dict:
    payload = {
        "model": model["id"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": TEMPERATURE,
        "usage": {"include": True},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "X-Title": "Compare AI meeting",
    }
    last_error = None
    for attempt in range(3):
        try:
            data = post_json(OPENROUTER_URL, payload, headers, REQUEST_TIMEOUT)
            break
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            last_error = f"HTTP {error.code}: {body[:300]}"
            if error.code not in (408, 429, 500, 502, 503, 504):
                raise RuntimeError(last_error)
            time.sleep(5 * (attempt + 1))
        except urllib.error.URLError as error:
            last_error = str(error)
            time.sleep(5 * (attempt + 1))
    else:
        raise RuntimeError(last_error or "запрос не прошёл")

    if "error" in data and not data.get("choices"):
        raise RuntimeError(json.dumps(data["error"], ensure_ascii=False)[:300])

    choice = (data.get("choices") or [{}])[0]
    content = (choice.get("message") or {}).get("content")
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    if not content:
        raise RuntimeError(f"пустой ответ, finish_reason={choice.get('finish_reason')}")

    usage = data.get("usage") or {}
    return {
        "text": content,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "cost": usage.get("cost"),
        "num_ctx": None,
        "context_limit": None,
        "truncated": data["choices"][0].get("finish_reason") == "length",
        "warning": None,
    }


def fetch_pricing() -> dict:
    try:
        request = urllib.request.Request("https://openrouter.ai/api/v1/models")
        with urllib.request.urlopen(request, timeout=30) as response:
            catalog = json.loads(response.read().decode("utf-8"))["data"]
    except Exception:
        return {}
    prices = {}
    for entry in catalog:
        pricing = entry.get("pricing") or {}
        try:
            prices[entry["id"]] = (float(pricing["prompt"]), float(pricing["completion"]))
        except (KeyError, TypeError, ValueError):
            continue
    return prices


def calc_cost(model_id: str, prompt_tokens, completion_tokens, prices: dict):
    if model_id not in prices or prompt_tokens is None or completion_tokens is None:
        return None
    price_in, price_out = prices[model_id]
    return prompt_tokens * price_in + completion_tokens * price_out


def check_protocol(text: str) -> dict:
    found = []
    for number, title in SECTION_TITLES.items():
        if re.search(rf"^#+\s*\**\s*{number}\.", text, re.MULTILINE) or title.lower() in text.lower():
            found.append(number)
    return {
        "chars": len(text),
        "sections": found,
        "missing": [n for n in SECTION_TITLES if n not in found],
        "utochnyty": len(re.findall(r"Уточнити", text, re.IGNORECASE)),
        "tables": text.count("| ---") + text.count("|---") + text.lower().count("<table"),
        "dates": len(set(re.findall(r"\b\d{2}[.\-/]\d{2}[.\-/]\d{4}\b", text))),
    }


def build_user_message(template: str, transcript: str, date: str, users: str) -> str:
    return (
        template.replace("[Date]", date)
        .replace("[Users]", users)
        .replace("[Transcription]", transcript)
    )


def run_model(model: dict, system: str, user: str, api_key: str | None) -> dict:
    started = time.time()
    try:
        if model["kind"] == "ollama":
            result = call_ollama(model, system, user)
        else:
            if not api_key:
                raise RuntimeError("нет ключа OPENROUTER_API_KEY")
            result = call_openrouter(model, system, user, api_key)
        result["error"] = None
    except Exception as error:
        result = {
            "text": "",
            "prompt_tokens": None,
            "completion_tokens": None,
            "cost": None,
            "num_ctx": None,
            "context_limit": None,
            "truncated": False,
            "warning": None,
            "error": str(error),
        }
    result["seconds"] = round(time.time() - started, 1)
    result["model"] = model
    result["checks"] = check_protocol(result["text"]) if result["text"] else None
    return result


def load_store(out_dir: Path) -> dict:
    path = out_dir / "runs.json"
    if path.exists():
        return json.loads(read_text(path))
    return {}


def save_store(out_dir: Path, store: dict) -> None:
    (out_dir / "runs.json").write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")


def write_outputs(out_dir: Path, transcript_name: str, transcript: str, store: dict) -> None:
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    good = []
    for label, meta in store.items():
        text_path = raw_dir / f"{label}.md"
        if meta.get("error") or not text_path.exists():
            continue
        good.append({"label": label, "meta": meta, "text": read_text(text_path)})
    random.shuffle(good)
    for blind, entry in zip(BLIND_LABELS, good):
        entry["blind"] = blind

    parts = [
        f"# Слепое сравнение протоколов: {transcript_name}",
        "",
        f"Сформировано {datetime.now():%d.%m.%Y %H:%M}. Протоколов на оценку: {len(good)}.",
        "",
        RUBRIC,
        "",
        "---",
        "",
        "## Стенограмма встречи",
        "",
        "```",
        transcript,
        "```",
        "",
    ]
    for result in good:
        parts += [
            "---",
            "",
            f"## Протокол {result['blind']}",
            "",
            result["text"],
            "",
        ]
    (out_dir / "for_opus.md").write_text("\n".join(parts), encoding="utf-8")

    key_lines = [
        f"# Ключ соответствия: {transcript_name}",
        "",
        "Открывать после того, как оценка выставлена.",
        "",
        "| Протокол | Модель | Слаг |",
        "|---|---|---|",
    ]
    for entry in sorted(good, key=lambda e: BLIND_LABELS.index(e["blind"])):
        key_lines.append(f"| {entry['blind']} | {entry['label']} | `{entry['meta']['model_id']}` |")
    (out_dir / "key.md").write_text("\n".join(key_lines) + "\n", encoding="utf-8")

    table = [
        f"# Результаты прогона: {transcript_name}",
        "",
        f"Дата прогона: {datetime.now():%d.%m.%Y %H:%M}. Температура {TEMPERATURE}, один прогон на модель,",
        "промпт и стенограмма одинаковые для всех.",
        "",
        "**Здесь видно, где какая модель. Открывать после того, как оценка по `for_opus.md` выставлена.**",
        "",
        f"| Модель | Токены вход | Токены выход | Стоимость, $ | Время, с | Символов | Разделов из {len(SECTION_TITLES)} | «Уточнити» | Обрезка |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    prices = fetch_pricing()
    texts = {entry["label"]: entry["text"] for entry in good}
    total_cost = 0.0
    byok = []
    for label, meta in store.items():
        if meta.get("error") or label not in texts:
            table.append(f"| {label} | ошибка: {str(meta.get('error'))[:80]} | | | {meta.get('seconds', '?')} | | | | |")
            continue
        checks = check_protocol(texts[label])
        cost = meta.get("cost")
        if meta.get("kind") == "openrouter":
            calculated = calc_cost(meta["model_id"], meta.get("prompt_tokens"), meta.get("completion_tokens"), prices)
            if calculated is not None and not cost:
                byok.append(label)
                cost = calculated
        total_cost += cost or 0.0
        table.append(
            "| {label} | {pin} | {pout} | {cost} | {sec} | {chars} | {sections} | {ut} | {trunc} |".format(
                label=label,
                pin=meta.get("prompt_tokens") or "?",
                pout=meta.get("completion_tokens") or "?",
                cost=f"{cost:.4f}" if cost is not None else "0 (локально)",
                sec=meta.get("seconds", "?"),
                chars=checks["chars"],
                sections=len(checks["sections"]),
                ut=checks["utochnyty"],
                trunc="да" if meta.get("truncated") else "нет",
            )
        )
    table += ["", f"Суммарная стоимость прогона: **${total_cost:.4f}**", ""]
    if byok:
        table += [
            "По моделям " + ", ".join(f"`{name}`" for name in byok) + " шлюз вернул стоимость 0: они ушли",
            "через BYOK и списались с вашего счёта OpenAI, а не с баланса OpenRouter. Для них в таблице",
            "стоимость посчитана по прайсу каталога.",
            "",
        ]

    problems = []
    for label, meta in store.items():
        if meta.get("error"):
            problems.append(f"- `{label}`: {meta['error']}")
            continue
        if meta.get("warning"):
            problems.append(f"- `{label}`: {meta['warning']}")
        if meta.get("truncated"):
            problems.append(f"- `{label}`: ответ или вход упёрлись в предел контекста")
        if label in texts:
            missing = check_protocol(texts[label])["missing"]
            if missing:
                problems.append(f"- `{label}`: нет разделов {', '.join(str(n) for n in missing)}")
    if problems:
        table += ["## На что обратить внимание", "", *problems, ""]
    (out_dir / "results.md").write_text("\n".join(table), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Сравнение моделей на протоколе встречи")
    parser.add_argument("--transcript", type=Path, default=DEFAULT_TRANSCRIPT)
    parser.add_argument("--models", help="список меток через запятую, например gpt-oss-20b,gpt-4.1-mini")
    parser.add_argument("--date", default=UNKNOWN_FIELD, help="что подставить в [Date] шаблона")
    parser.add_argument("--users", default=UNKNOWN_FIELD, help="что подставить в [Users] шаблона")
    parser.add_argument("--list", action="store_true", help="показать кандидатов и выйти")
    parser.add_argument("--rebuild", action="store_true", help="пересобрать файлы из runs.json, без запросов к моделям")
    args = parser.parse_args()

    if args.rebuild:
        out_dir = RESULTS_DIR / args.transcript.stem
        store = load_store(out_dir)
        if not store:
            print(f"Нечего пересобирать: нет {out_dir / 'runs.json'}")
            return 1
        transcript = read_text(args.transcript).strip()
        write_outputs(out_dir, args.transcript.stem, transcript, store)
        print(f"Пересобрано из {len(store)} записей: {out_dir}")
        return 0

    if args.list:
        for model in MODELS:
            state = "вкл" if model["enabled"] else "выкл"
            print(f"{model['label']:20} {model['kind']:12} {model['id']:32} [{state}]")
        return 0

    selected = [m for m in MODELS if m["enabled"]]
    if args.models:
        wanted = {name.strip() for name in args.models.split(",")}
        selected = [m for m in MODELS if m["label"] in wanted]
        unknown = wanted - {m["label"] for m in MODELS}
        if unknown:
            print(f"Неизвестные метки: {', '.join(sorted(unknown))}")
            return 1
    if not selected:
        print("Не выбрано ни одной модели")
        return 1

    if not args.transcript.exists():
        print(f"Нет файла стенограммы: {args.transcript}")
        return 1
    for path in (SYSTEM_PROMPT_PATH, USER_PROMPT_PATH):
        if not path.exists():
            print(f"Нет файла промпта: {path}")
            return 1

    system = read_text(SYSTEM_PROMPT_PATH).strip()
    transcript = read_text(args.transcript).strip()
    user = build_user_message(read_text(USER_PROMPT_PATH).strip(), transcript, args.date, args.users)
    api_key = load_api_key()
    if not api_key and any(m["kind"] == "openrouter" for m in selected):
        print("Нет ключа: искал", " и ".join(KEY_NAMES), "в", ", ".join(str(p) for p in ENV_FILES), "и в окружении")
        return 1

    print(f"Стенограмма: {args.transcript.name}, {len(transcript)} символов, ~{estimate_tokens(transcript)} токенов")
    print(f"Промпты: {SYSTEM_PROMPT_PATH.name} {len(system)} символов, {USER_PROMPT_PATH.name} с датой «{args.date}», участниками «{args.users}»")
    print(f"Моделей в прогоне: {len(selected)}\n")

    out_dir = RESULTS_DIR / args.transcript.stem
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    store = load_store(out_dir)

    for model in selected:
        print(f"[{model['label']}] запуск...", flush=True)
        result = run_model(model, system, user, api_key)
        if result["error"]:
            print(f"[{model['label']}] ошибка: {result['error']}")
        else:
            cost = result["cost"]
            money = f", ${cost:.4f}" if cost else ""
            print(f"[{model['label']}] готово за {result['seconds']} с, {len(result['text'] or '')} символов{money}")
            if result["warning"]:
                print(f"[{model['label']}] внимание: {result['warning']}")
        if result["text"]:
            (raw_dir / f"{model['label']}.md").write_text(result["text"], encoding="utf-8")
        store[model["label"]] = {
            "model_id": model["id"],
            "kind": model["kind"],
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "cost": result["cost"],
            "seconds": result["seconds"],
            "truncated": result["truncated"],
            "warning": result["warning"],
            "error": result["error"],
            "ran_at": f"{datetime.now():%d.%m.%Y %H:%M}",
        }

    save_store(out_dir, store)
    write_outputs(out_dir, args.transcript.stem, transcript, store)
    print(f"\nГотово. Файлы в {out_dir}")
    print("  for_opus.md  несите в веб-интерфейс Claude, там протоколы обезличены")
    print("  results.md   цифры и имена моделей, открывать после оценки")
    print("  key.md       ключ соответствия А/Б/В/Г")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
