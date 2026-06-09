// Импортируемые помощники «теорема/определение» (офлайн, без внешних пакетов).
// Импортируется и в main.typ, и во фрагменты, и в проверочные обёртки:
//   #import "_preamble.typ": *
// НЕ менять сигнатуры: theorem(title: none, body), definition(title: none, body).

#let theorem(title: none, body) = block(width: 100%, inset: 10pt, radius: 4pt,
  fill: rgb("#eef3ff"), stroke: 0.5pt + rgb("#9bb4e8"),
  [*Теорема#if title != none [ (#title)]:* #body])

#let definition(title: none, body) = block(width: 100%, inset: 10pt, radius: 4pt,
  fill: rgb("#f3f3f3"), [*Определение#if title != none [ (#title)]:* #body])
