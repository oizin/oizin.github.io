<!--
Add here global page variables to use throughout your
website.
The website_* must be defined for the RSS to work
-->
@def website_title = "Oisin Fitzgerald"
@def website_descr = "My personal website"
@def website_url   = "https://oizin.github.io"

@def author = "Oisin Fitzgerald"

@def mintoclevel = 2

@def generate_rss = true
@def rss_website_title = "Oisin Fitzgerald"
@def rss_website_descr = "My personal website"
@def rss_website_url   = "https://oizin.github.io"
@def rss_full_content = true

<!--
Add here files or directories that should be ignored by Franklin, otherwise
these files might be copied and, if markdown, processed by Franklin which
you might not want. Indicate directories by ending the name with a `/`.
-->
@def ignore = ["node_modules/", "franklin", "franklin.pub"]

<!--
Add here global latex commands to use throughout your
pages. It can be math commands but does not need to be.
For instance:
* \newcommand{\phrase}{This is a long phrase to copy.}
-->
\newcommand{\R}{\mathbb R}
\newcommand{\scal}[1]{\langle #1 \rangle}
