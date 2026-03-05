import { useEffect, useMemo, useState } from 'react'
import axios from 'axios'
import * as Checkbox from '@radix-ui/react-checkbox'
import * as Label from '@radix-ui/react-label'
import * as ScrollArea from '@radix-ui/react-scroll-area'
import * as Separator from '@radix-ui/react-separator'

type Locale = 'en' | 'km'

type Symptom = {
  code: string
  label: string
  label_en: string
  label_km: string
}

type Prediction = {
  condition: string
  condition_label: string
  probability: number
}

type PredictResponse = {
  locale: Locale
  predicted_condition: string
  predicted_condition_label: string
  confidence: number
  top_predictions: Prediction[]
  active_symptom_count: number
}

const apiBase = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const localeStorageKey = 'app.locale'

const toLocale = (value: string | null | undefined): Locale => {
  if (!value) return 'en'
  return value.toLowerCase().startsWith('km') ? 'km' : 'en'
}

const messages = {
  en: {
    language: 'Language',
    english: 'English',
    khmer: 'Khmer',
    appTitle: 'Medical Symptom Classifier',
    appSubtitle: 'Select symptoms and run a prediction using your trained multiclass model.',
    searchSymptoms: 'Search Symptoms',
    searchPlaceholder: 'Type fatigue, jaundice, cough...',
    symptomsVisible: 'symptoms visible',
    selected: 'selected',
    predictCondition: 'Predict Condition',
    predicting: 'Predicting...',
    predictionResult: 'Prediction Result',
    predictionSubtitle: 'Top ranked conditions from the trained model.',
    failedToLoadSymptoms: 'Failed to load symptoms from API. Is backend running on port 8000?',
    selectAtLeastOne: 'Please choose at least one symptom.',
    predictionFailed: 'Prediction failed. Please try again.',
    noPredictionYet: 'No prediction yet. Choose symptoms and press Predict Condition.',
    mostLikely: 'Most likely condition',
    confidence: 'Confidence',
    topPredictions: 'Top 3 Predictions',
  },
  km: {
    language: 'ភាសា',
    english: 'អង់គ្លេស',
    khmer: 'ខ្មែរ',
    appTitle: 'ប្រព័ន្ធទស្សន៍ទាយជំងឺតាមរោគសញ្ញា',
    appSubtitle: 'ជ្រើសរោគសញ្ញា ហើយចុចទស្សន៍ទាយពីម៉ូដែលដែលបានបណ្តុះបណ្តាល។',
    searchSymptoms: 'ស្វែងរករោគសញ្ញា',
    searchPlaceholder: 'វាយពាក្យដូចជា អស់កម្លាំង លឿង ក្អក...',
    symptomsVisible: 'រោគសញ្ញាដែលបង្ហាញ',
    selected: 'បានជ្រើស',
    predictCondition: 'ទស្សន៍ទាយជំងឺ',
    predicting: 'កំពុងទស្សន៍ទាយ...',
    predictionResult: 'លទ្ធផលទស្សន៍ទាយ',
    predictionSubtitle: 'លទ្ធផលជំងឺដែលមានលទ្ធភាពខ្ពស់ពីម៉ូដែល។',
    failedToLoadSymptoms: 'មិនអាចទាញយករោគសញ្ញាពី API បានទេ។ សូមពិនិត្យ backend នៅ port 8000។',
    selectAtLeastOne: 'សូមជ្រើសរោគសញ្ញាយ៉ាងហោចណាស់មួយ។',
    predictionFailed: 'ការទស្សន៍ទាយបរាជ័យ។ សូមព្យាយាមម្តងទៀត។',
    noPredictionYet: 'មិនទាន់មានលទ្ធផលទស្សន៍ទាយទេ។ សូមជ្រើសរោគសញ្ញា ហើយចុចទស្សន៍ទាយ។',
    mostLikely: 'ជំងឺដែលមានលទ្ធភាពខ្ពស់បំផុត',
    confidence: 'ទំនុកចិត្ត',
    topPredictions: 'លទ្ធផលកំពូល 3',
  },
} as const

const normalizeSearchText = (value: string): string =>
  value
    .toLowerCase()
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[_-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()

const fuzzyScore = (query: string, candidate: string): number | null => {
  const q = normalizeSearchText(query)
  const c = normalizeSearchText(candidate)

  if (!q || !c) return null
  if (c.includes(q)) {
    const start = c.indexOf(q)
    const wordBoundaryBonus = start === 0 || c[start - 1] === ' ' ? 0.5 : 0
    return 100 - start * 0.5 - (c.length - q.length) * 0.05 + wordBoundaryBonus
  }

  let lastIndex = -1
  let firstMatch = -1
  let contiguous = 0
  let boundaryHits = 0

  for (const ch of q) {
    const index = c.indexOf(ch, lastIndex + 1)
    if (index === -1) return null
    if (firstMatch === -1) firstMatch = index
    if (index === lastIndex + 1) contiguous += 1
    if (index === 0 || c[index - 1] === ' ') boundaryHits += 1
    lastIndex = index
  }

  const spread = lastIndex - firstMatch + 1
  return q.length * 4 + contiguous * 1.5 + boundaryHits - (spread - q.length) * 0.35 - c.length * 0.02
}

export default function App() {
  const [symptoms, setSymptoms] = useState<Symptom[]>([])
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [locale, setLocale] = useState<Locale>(() => toLocale(localStorage.getItem(localeStorageKey)))
  const text = messages[locale]

  useEffect(() => {
    localStorage.setItem(localeStorageKey, locale)
    document.documentElement.lang = locale
  }, [locale])

  useEffect(() => {
    setError(null)
    setResult(null)
    axios
      .get<{ locale: Locale; symptoms: Symptom[] }>(`${apiBase}/symptoms`, { params: { lang: locale } })
      .then((res) => setSymptoms(res.data.symptoms))
      .catch(() => setError(text.failedToLoadSymptoms))
  }, [locale, text.failedToLoadSymptoms])

  const filteredSymptoms = useMemo(() => {
    const needle = query.trim()
    if (!needle) return symptoms

    return symptoms
      .map((symptom) => {
        const candidates = [symptom.label, symptom.label_en, symptom.label_km, symptom.code]
        const score = Math.max(...candidates.map((candidate) => fuzzyScore(needle, candidate) ?? -Infinity))
        return { symptom, score }
      })
      .filter((item) => Number.isFinite(item.score))
      .sort((a, b) => b.score - a.score || a.symptom.label.localeCompare(b.symptom.label))
      .map((item) => item.symptom)
  }, [query, symptoms])

  const toggleSymptom = (code: string) => {
    const next = new Set(selected)
    if (next.has(code)) {
      next.delete(code)
    } else {
      next.add(code)
    }
    setSelected(next)
  }

  const onPredict = async () => {
    setError(null)
    setResult(null)

    if (selected.size === 0) {
      setError(text.selectAtLeastOne)
      return
    }

    try {
      setLoading(true)
      const payload = { symptoms: Array.from(selected) }
      const res = await axios.post<PredictResponse>(`${apiBase}/predict`, payload, { params: { lang: locale } })
      setResult(res.data)
    } catch (e: any) {
      const msg = e?.response?.data?.detail || text.predictionFailed
      setError(String(msg))
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="mx-auto max-w-6xl px-6 py-10">
      <div className="grid gap-8 lg:grid-cols-[1.35fr_1fr]">
        <section className="rounded-2xl border border-brand-100 bg-white/80 p-6 shadow-sm backdrop-blur">
          <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
            <h1 className="text-3xl font-extrabold text-slate-900">{text.appTitle}</h1>
            <div className="flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-2 py-1">
              <span className="text-xs font-semibold text-slate-500">{text.language}</span>
              <button
                onClick={() => setLocale('en')}
                className={`rounded px-2 py-1 text-xs font-semibold transition ${
                  locale === 'en' ? 'bg-brand-700 text-white' : 'text-slate-600 hover:bg-slate-100'
                }`}
              >
                {text.english}
              </button>
              <button
                onClick={() => setLocale('km')}
                className={`rounded px-2 py-1 text-xs font-semibold transition ${
                  locale === 'km' ? 'bg-brand-700 text-white' : 'text-slate-600 hover:bg-slate-100'
                }`}
              >
                {text.khmer}
              </button>
            </div>
          </div>
          <p className="mt-2 text-slate-600">
            {text.appSubtitle}
          </p>

          <div className="mt-6">
            <Label.Root htmlFor="search" className="text-sm font-semibold text-slate-700">
              {text.searchSymptoms}
            </Label.Root>
            <input
              id="search"
              className="mt-2 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm outline-none ring-brand-400 transition focus:ring"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={text.searchPlaceholder}
            />
          </div>

          <div className="mt-4 flex items-center justify-between text-sm text-slate-600">
            <span>
              {filteredSymptoms.length} {text.symptomsVisible}
            </span>
            <span>
              {selected.size} {text.selected}
            </span>
          </div>

          <ScrollArea.Root className="mt-4 h-[420px] overflow-hidden rounded-xl border border-slate-200 bg-slate-50">
            <ScrollArea.Viewport className="h-full w-full p-4">
              <div className="grid gap-2 sm:grid-cols-2">
                {filteredSymptoms.map((symptom) => (
                  <label
                    key={symptom.code}
                    className="flex cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 hover:bg-white"
                  >
                    <Checkbox.Root
                      checked={selected.has(symptom.code)}
                      onCheckedChange={() => toggleSymptom(symptom.code)}
                      className="h-4 w-4 rounded border border-slate-400 bg-white data-[state=checked]:border-brand-700 data-[state=checked]:bg-brand-700"
                    >
                      <Checkbox.Indicator className="block text-center text-xs font-bold text-white">
                        ✓
                      </Checkbox.Indicator>
                    </Checkbox.Root>
                    <span className="text-sm text-slate-700">{symptom.label}</span>
                  </label>
                ))}
              </div>
            </ScrollArea.Viewport>
            <ScrollArea.Scrollbar className="flex touch-none select-none bg-slate-100 p-0.5" orientation="vertical">
              <ScrollArea.Thumb className="relative flex-1 rounded-full bg-slate-300" />
            </ScrollArea.Scrollbar>
          </ScrollArea.Root>

          <button
            onClick={onPredict}
            disabled={loading}
            className="mt-5 w-full rounded-lg bg-brand-700 px-4 py-2.5 font-semibold text-white transition hover:bg-brand-800 disabled:cursor-not-allowed disabled:opacity-70"
          >
            {loading ? text.predicting : text.predictCondition}
          </button>
        </section>

        <section className="rounded-2xl border border-slate-200 bg-white/85 p-6 shadow-sm backdrop-blur">
          <h2 className="text-xl font-bold text-slate-900">{text.predictionResult}</h2>
          <p className="mt-1 text-sm text-slate-600">{text.predictionSubtitle}</p>

          <Separator.Root className="my-5 h-px bg-slate-200" />

          {error && <div className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">{error}</div>}

          {!error && !result && (
            <div className="rounded-lg border border-dashed border-slate-300 p-4 text-sm text-slate-600">
              {text.noPredictionYet}
            </div>
          )}

          {result && (
            <div className="space-y-4">
              <div className="rounded-xl border border-brand-200 bg-brand-50 p-4">
                <div className="text-sm text-brand-900">{text.mostLikely}</div>
                <div className="mt-1 text-2xl font-extrabold text-brand-900">
                  {result.predicted_condition_label || result.predicted_condition}
                </div>
                <div className="mt-1 text-sm text-brand-800">
                  {text.confidence}: {(result.confidence * 100).toFixed(2)}%
                </div>
              </div>

              <div>
                <h3 className="text-sm font-semibold text-slate-700">{text.topPredictions}</h3>
                <div className="mt-2 space-y-2">
                  {result.top_predictions.map((pred) => (
                    <div key={pred.condition} className="rounded-lg border border-slate-200 p-3">
                      <div className="flex items-center justify-between">
                        <span className="font-medium text-slate-800">{pred.condition_label || pred.condition}</span>
                        <span className="text-sm text-slate-600">{(pred.probability * 100).toFixed(2)}%</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </section>
      </div>
    </main>
  )
}
