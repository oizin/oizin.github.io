@def title = "[notes]"

\newcommand{\figenv}[3]{
~~~
<figure style="text-align:center;padding:0;margin:0">
<img src="!#2" style="padding:0;border:1px solid black;margin:0;#3" alt="#1"/>
<figcaption>#1</figcaption>
</figure>
~~~
}

\figenv{}{/assets/banner.png}{width:100%}
