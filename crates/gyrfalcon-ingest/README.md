# gyrfalcon-ingest

Turns PDF, HTML, and plain text into text for a Gyrfalcon collection. PDF goes through PDFium, loaded at run time, with a layout layer that drops running headers and rejoins words split across lines.

The search engine itself is the `gyrfalcon` crate. A program that only indexes text it already has does not need this crate.

Repository: <https://github.com/lucasdmarshall/gyrfalcon>
