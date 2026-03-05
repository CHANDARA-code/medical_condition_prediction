import { useEffect, useMemo, useState } from 'react'
import axios from 'axios'
import * as Checkbox from '@radix-ui/react-checkbox'
import * as Label from '@radix-ui/react-label'
import * as ScrollArea from '@radix-ui/react-scroll-area'
import * as Separator from '@radix-ui/react-separator'

type Symptom = {
  code: string
  label: string
}

type Prediction = {
  condition: string
  probability: number
}

type PredictResponse = {
  predicted_condition: string
  confidence: number
  top_predictions: Prediction[]
  active_symptom_count: number
}

const apiBase = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export default function App() {
  const [symptoms, setSymptoms] = useState<Symptom[]>([])
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    axios
      .get<{ symptoms: Symptom[] }>(`${apiBase}/symptoms`)
      .then((res) => setSymptoms(res.data.symptoms))
      .catch(() => setError('Failed to load symptoms from API. Is backend running on port 8000?'))
  }, [])

  const filteredSymptoms = useMemo(() => {
    const needle = query.trim().toLowerCase()
    if (!needle) return symptoms
    return symptoms.filter(
      (s) => s.code.includes(needle) || s.label.toLowerCase().includes(needle),
    )
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
      setError('Please choose at least one symptom.')
      return
    }

    try {
      setLoading(true)
      const payload = { symptoms: Array.from(selected) }
      const res = await axios.post<PredictResponse>(`${apiBase}/predict`, payload)
      setResult(res.data)
    } catch (e: any) {
      const msg = e?.response?.data?.detail || 'Prediction failed. Please try again.'
      setError(String(msg))
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="mx-auto max-w-6xl px-6 py-10">
      <div className="grid gap-8 lg:grid-cols-[1.35fr_1fr]">
        <section className="rounded-2xl border border-brand-100 bg-white/80 p-6 shadow-sm backdrop-blur">
          <h1 className="text-3xl font-extrabold text-slate-900">Medical Symptom Classifier</h1>
          <p className="mt-2 text-slate-600">
            Select symptoms and run a prediction using your trained multiclass model.
          </p>

          <div className="mt-6">
            <Label.Root htmlFor="search" className="text-sm font-semibold text-slate-700">
              Search Symptoms
            </Label.Root>
            <input
              id="search"
              className="mt-2 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm outline-none ring-brand-400 transition focus:ring"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Type fatigue, jaundice, cough..."
            />
          </div>

          <div className="mt-4 flex items-center justify-between text-sm text-slate-600">
            <span>{filteredSymptoms.length} symptoms visible</span>
            <span>{selected.size} selected</span>
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
            {loading ? 'Predicting...' : 'Predict Condition'}
          </button>
        </section>

        <section className="rounded-2xl border border-slate-200 bg-white/85 p-6 shadow-sm backdrop-blur">
          <h2 className="text-xl font-bold text-slate-900">Prediction Result</h2>
          <p className="mt-1 text-sm text-slate-600">Top ranked conditions from the trained model.</p>

          <Separator.Root className="my-5 h-px bg-slate-200" />

          {error && <div className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">{error}</div>}

          {!error && !result && (
            <div className="rounded-lg border border-dashed border-slate-300 p-4 text-sm text-slate-600">
              No prediction yet. Choose symptoms and press Predict Condition.
            </div>
          )}

          {result && (
            <div className="space-y-4">
              <div className="rounded-xl border border-brand-200 bg-brand-50 p-4">
                <div className="text-sm text-brand-900">Most likely condition</div>
                <div className="mt-1 text-2xl font-extrabold text-brand-900">{result.predicted_condition}</div>
                <div className="mt-1 text-sm text-brand-800">
                  Confidence: {(result.confidence * 100).toFixed(2)}%
                </div>
              </div>

              <div>
                <h3 className="text-sm font-semibold text-slate-700">Top 3 Predictions</h3>
                <div className="mt-2 space-y-2">
                  {result.top_predictions.map((pred) => (
                    <div key={pred.condition} className="rounded-lg border border-slate-200 p-3">
                      <div className="flex items-center justify-between">
                        <span className="font-medium text-slate-800">{pred.condition}</span>
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
