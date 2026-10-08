/** What the bible's grid and its editor both read off a production's members. */
import type { MemberInfo, Production, ShotInfo } from './api'
import { FINISHED, GENERATED, castOf, pad } from './production'

export const KIND_LABEL: Record<string, string> = {
  person: 'CHARACTER',
  animal: 'CHARACTER',
  environment: 'LOCATION',
  object: 'PROP',
  style: 'STYLE',
}

export const TABS: { id: string; kinds: string[] }[] = [
  { id: 'all', kinds: [] },
  { id: 'characters', kinds: ['person', 'animal'] },
  { id: 'locations', kinds: ['environment'] },
  { id: 'props', kinds: ['object'] },
  { id: 'style', kinds: ['style'] },
]

export const filmOf = (p: Production): ShotInfo[] => p.chapters.flatMap((c) => c.shots)
export const casts = (shot: ShotInfo, slug: string) => shot.castings.some((c) => c.member === slug) || castOf(shot).includes(slug)
export const linesOf = (shot: ShotInfo, slug: string) =>
  ((shot.plan.dialogue as { who: string }[] | undefined) ?? []).filter((d) => d.who === slug).length
export const shotsOf = (p: Production, slug: string) => filmOf(p).filter((s) => casts(s, slug))
export const shotLabel = (shot: ShotInfo) => `${pad(shot.chapter + 1, 2)}.${pad(shot.position + 1, 3)}`

type Ref = MemberInfo['references'][number]
const shown = (r: Ref) => !!r.artifact && r.present
export const photographs = (m: MemberInfo) => m.references.filter((r) => r.role !== GENERATED && shown(r))
export const drawnRef = (m: MemberInfo) => m.references.find((r) => r.role === GENERATED)

/** "2 photos · 1 generated", "drawn from words", "no reference". */
export function source(m: MemberInfo): { label: string; provided: boolean } {
  const photos = new Set(photographs(m).map((r) => r.photo)).size
  const generated = m.references.filter((r) => r.role === GENERATED && shown(r)).length
  if (!photos) return { label: generated ? 'drawn from words' : 'no reference', provided: false }
  const parts = [`${photos} photo${photos === 1 ? '' : 's'}`, ...(generated ? [`${generated} generated`] : [])]
  return { label: parts.join(' · '), provided: true }
}

/** The picture a card leads with: a whole view before a drawing before a face. */
export function pictures(m: MemberInfo): Ref[] {
  const rank = (r: Ref) => (r.role === 'face' ? 2 : r.role === GENERATED ? 1 : 0)
  return m.references.filter(shown).sort((a, b) => rank(a) - rank(b))
}

export const traitsOf = (m: MemberInfo) => m.identity.traits ?? []
/** Its look in words: as a person wrote it, or its traits joined as the card says them. */
export const appearance = (m: MemberInfo) => m.appearance || traitsOf(m).map((t) => t.phrase).join(', ')

/** The meta line under a card's words, per kind. */
export function meta(p: Production, m: MemberInfo): { k: string; v: string } {
  if (m.kind === 'person' || m.kind === 'animal') return { k: 'VOICE', v: [m.voice, m.delivery].filter(Boolean).join(', ') || 'not set: its noun’s natural voice' }
  if (m.kind === 'environment') return { k: 'LIGHTS', v: (m.identity.lights ?? []).join(', ') || '—' }
  if (m.kind === 'style') return { k: 'INJECTED INTO', v: 'every Qwen-Image prompt' }
  const chapters = [...new Set(shotsOf(p, m.slug).map((s) => pad(s.chapter + 1, 2)))]
  return { k: 'IN', v: chapters.length ? `ch ${chapters.join(', ')}` : 'no shot yet' }
}

export interface StateRow {
  id: number | null
  label: string
  parent: string
  amend: string
  shot: ShotInfo | undefined
  image: string
  artifact: string | null
}

/** Each look a shot gives the member, in film order, with the look it was drawn from. */
export function states(p: Production, m: MemberInfo): StateRow[] {
  const byId = new Map(filmOf(p).map((s) => [s.id, s]))
  return (m.looks ?? []).map((l) => ({
    id: l.id,
    label: l.label,
    parent: l.source.id === null ? 'base' : l.source.label,
    amend: l.amend,
    shot: l.shot === null ? undefined : byId.get(l.shot),
    image: l.image,
    artifact: l.artifact,
  }))
}

/** "CHANGES IN CH 01", or across every chapter its changes are in. */
export function changesIn(rows: StateRow[]): string {
  const chapters = [...new Set(rows.flatMap((r) => (r.shot ? [pad(r.shot.chapter + 1, 2)] : [])))]
  return chapters.length === 1 ? `CHANGES IN CH ${chapters[0]}` : `CHANGES IN CH ${chapters.join(', ')}`
}

export const done = (thing: { state: string }) => FINISHED.has(thing.state)

/** "3 h 46 min", "12 min", "40 s". */
export function hours(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)} s`
  const minutes = Math.round(seconds / 60)
  return minutes < 60 ? `${minutes} min` : `${Math.floor(minutes / 60)} h ${minutes % 60} min`
}

/** What an edit makes stale and what redoing it costs, by this machine's measured timings. */
export function impact(p: Production, slug: string, voiceOnly: boolean) {
  const t = p.timings ?? {}
  const judge = t.judge?.seconds ?? 0
  const clipCost = (s: ShotInfo) => (t.render_shot?.per_second ? t.render_shot.per_second * s.seconds : (t.render_shot?.seconds ?? 0)) + judge
  const shots = shotsOf(p, slug).filter((s) => !voiceOnly || linesOf(s, slug) > 0)
  const keyframes = voiceOnly ? [] : shots.filter((s) => s.keyframe.source === 'composed' && done(s.keyframe))
  const clips = shots.filter((s) => done(s.clip))
  const seconds = keyframes.length * ((t.compose_keyframe?.seconds ?? 0) + judge) + clips.reduce((n, s) => n + clipCost(s), 0)
  return { keyframes: keyframes.length, clips: clips.length, seconds }
}
