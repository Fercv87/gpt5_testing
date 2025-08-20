# gpt5_testing
This repo holds the outputs of testing GPT5.

This is the supporting repo for the Medium Article: https://medium.com/@fercv87/testing-gpt-5-with-financial-regulation-d8f1d2bc6736

Introduction
Why this article may be worth reading? Because I deep dive in a specific domain (i.e. financial regulation) to test the capabilities of GPT-5. All testing outputs are in GitHub.

Two unrelated publications happened in the last days:

The European Central Bank (ECB) published an update of its guide to Internal Models.
Open AI released GPT-5 its new flagship model.
The aim of this article is to blend those two events together. How? I’m going to test GPT-5 capabilities using the ECB´s revised guide to internal models as an excuse.

Context on the functional domain
First, let´s give context on the specific functional domain we are using as playground. For those unfamiliar with Financial Regulation, that guide sets out the ECB´s understanding of the applicable rules for European Financial Institutions in the context of the regulatory capital requirements calculation under CRR3. In a nutshell, how the ECB understands and applies key regulation while performing its supervisory mandate.

The Guide to Internal Models is an extensive and thorough document (300+ pages). It breaks down the requirements to develop, validate, deploy and use internal models (such as Probability of Default or Loss Given Default). It set out the expectations around the end-to-end of the model life cycle management and the three lines of defense involved in it. Its scope is so broad that it addresses topics such as data governance & data management and the implementation of the models in the financial institution`s IT infrastructure.

Indeed, this revised guide addresses, for the first time, the usage of machine learning techniques. This is how ECB defines that: “ML techniques refer to models which rely on a large number of parameters, can represent highly non-linear functions and/or may require large amounts of data for their calibration (training).” ECB clarifies ML techniques do not include generalised linear models (such as linear or logistic regressions).

First test: PDF to JSON parsing
Now, let´s jump to the GPT-5. In the past, I´ve tried and failed to programmatically extract text from a PDF document and parse it into a JSON file with a structure that for later use enables traceability to the source. This is a key requirement for a reasonable business usage. It took me some manual effort to put together a JSON file for the AI Act. The first test is to see whether GPT-5 can do so. I added the PDF raw file and three screenshots with my prompt:

< I want you to be an expert reading PDF documents and transforming them into JSON files. I want you to transform the attached PDF file into a JSON file. The structure shall be following, see an example I´ve manually created:

<[{ “title”: “Foreword”,

“subtitle”: N/A,

“paragraph_number”: “1”,

“page”: 5,

“text”: “Articles 143, 283 and …” }, ….>

This should be the structure for each of the paragraphs that in the PDF are flagged with numbers.

In the two screenshots I share here, first you can see from where to take “title” and “subtitle”.

For those two items it could be there is no “subtitle” as it happens for “title”: “Foreword”.

But for instance, for the “title”: “Overarching principles for internal models” there are different subtitles: “Guidelines at consolidated and subsidiary levels”, “Documentation of internal models”,…

Another important point is what I was telling you before about the paragraphs. The document itself labels the paragraphs with numbers, that drives the logic to inform “paragraph_number” and “text”.

Disregard the text in the footnotes and tables in the document (see the third screenshot) to understand those tables shall be excluded from the JSON file, also from the second screenshot you can see the footnotes shall be excluded as well.

As such the JSON file content in the “text” shall start from page 5 to page 357. Transpose all that text into the JSON file.

Do you very best effort in reasoning and execution capabilities.

It´s essential that the JSON file is complete.

My project depends on this task and it´s crucial.

Take all the time and resources you may need. >


GPT-5 works fine a multi-modal approach. Screenshots and uploaded documents tend to enrich the context given it has 400k token context window.
After 5 minutes I got the first JSON file ready to download. The first iteration was partially successful as the “title” and “subtitle” were not captured properly. It offered me the possibility to add an extra level. That was something I didn´t notice, but the suggestion made complete sense. I went for that option and iterated 2 more times where I showed the output I was getting and what I was expecting.

Press enter or click to view image in full size

It took about 5 minutes to get the first version of the ready to download JSON file.
On top of getting the JSON, I asked for the code. 200 lines of Python code that I tested with the previous ECB´s guide from February 2024, getting positive results.

Second test: coding for NLP with NLTK
Next was to check how does GPT-5 perform coding. The goal is to get a WordCloud on the two versions of the ECB´s guide to internal models. After reading two must-have books: The Pragmatic Programmer and A philosophy of software design, I´ve been improving my prompts where the expected output is Python code. Below there are three prompts, the first one is the standard for the use case of coding. The second one is the first iteration aiming to get a quick solution. And the third one is a second iteration with the refactorization using functions and best coding practices of the previous block of code. The firs iteration was successful straightaway. The second iteration aiming to refactor code with functions was not successful merely because it took several attempts to fine tune the look and feel of the WordCloud.

First prompt:

< You are the “Expert Python Support Assistant”: a prompt‐driven helper specialized in solving Python problems and teaching via clear, idiomatic, and runnable code. Also behave as an expert in NLP tasks and NLTK library and Wordcloud.

Also follow these guidelines:

<### Core Guidelines & Best Practices –

- **Correctness > Clarity > Brevity**

- **PEP 8 Compliant**: snake_case for variables/functions, PascalCase for classes; 79-char line length.

- **Meaningful Names**: avoid foo, tmp; choose descriptive names.

- **Single Responsibility**: one clear purpose per function/class.

- **DRY**: factor and reuse shared logic.

- **Modular Code**: group related functions into modules; use modules as singletons for shared utilities.

- **Explicitness Over Cleverness**: no magic numbers; clear, step-by-step logic.

- **Type Annotations**: annotate all public functions/methods.

- **Fail-Early Validation**: raise specific exceptions on invalid input.

- **Handle Exceptions Specifically**: catch only the exceptions you expect.

- **Context Managers**: use with or @contextmanager for resource management.

- **Comprehensions & Generators**: favor list/dict/set comprehensions and generator expressions for concise, efficient iteration.

- **Avoid Global State**: prefer function args, return values, or object attributes.

- **Docstrings & Comments**: document every public function/class; keep comments concise and relevant.

- **Unit Testing**: design code that’s easy to test; include simple tests when appropriate.

- **Minimal Dependencies**: stick to the standard library unless an external library is clearly justified.

- **Virtual Environments**: presume the user manages dependencies in a venv; mention only if setup is relevant.>>

Second prompt:

< I want to get a Word Cloud of the attached JSON file that has this structure:

“[{ “title”: “Foreword”,

“section”: “N/A”,

“subsection”: “N/A”,

“paragraph_number”: “1”,

“page”: 4,

“text”: “Articles 143, 283 and 363 of Regulation (EU) ….” },”

The actual content is in the block “text”.

I´m not sure whether it would be better to directly target the PDF. Just let me know about the best strategy.

Give a script that is not a function as I want the bare minimum lines of code. But give me that script with all the important arguments for parameterisation of the NLTK functions I will be using.

I stress your python code shall avoid functions that add clutter.

Do you very best effort in reasoning and execution capabilities. Take all the time and resources you may need.>

Third prompt:

< Now follow best coding practices and translate this code into different manageable functions:

<code>

Recall following:

<same guidelines from the first prompt>

In each function comment, add a WHY the function is needed and WHAT it does very briefly.>

Third test: Multi-Agent-System powered by GPT-5.
The third test went one step further. Testing the model via API. In a sort of recursive fashion, I´m working in an accelerator Multi-Agent-System (MAS) that develops other MAS with specific purposes and for narrow use cases. The goal was to power the Accelerator MAS with GPT-5-mini. Using a simple example for the Accelerator MAS to develop a Legal Analyst MAS. It took around ten minutes of inference and less than 0.2$.

Lang Smith project trace of the Legal Analyst MAS.
However, the example should have been much more precise on the expected outcome, formatting and goal of this Legal Analyst MAS. It took some further iterations editing the Legal Analyst MAS to get the desired outcome. The second iteration, using GPT-5, resulted in an acceptable outcome.

As there was room for improvement, the Legal Analyst MAS run a full comparison, using again GPT-5, of the two documents yielding a table at paragraph level . That increased the cost to almost 4$. But the outcome was definitely worth it.

The target outcome was a table like output that, first, keeps an explicit traceability to the original documents; second, adds the actual original documents excerpts; third, provides a high-level summary of the differences between the two documents.

At this junction, the reader may notice the two previous tests become truly purposeful and meaningful.

For a professional comparison of two legal documents (one deprecating the other), you want the possibility to go back to the actual document to double check there are no loose ties. For a faster review, you want the literals of the original document but formatted in a palatable way. And finally, you want to leverage on GenAI to get an intuition of the differences between the two documents.

All of that is granted if you captured in the JSON file the structure of the document. Additionally, visualizing with a WordCloud the extracted text helps you confirm everything went fine. In a professional environment, I would do further checks on the text (n-grams, statistics on #characters, #words and #sentences, random manual checks across the document,…).

Conclusion
So far, I cannot complain about GPT-5 performance in the use cases I´ve tested (coding and debugging). The bigger context window is a material improvement. Small tip, force the router to the reasoning and most powerful underpinning model by adding in your prompts something like “Do you very best effort in reasoning and execution capabilities”.

At the end, this is a very powerful tool and used properly yields excellent outcomes. As it happens in other domains, such as cooking, a knife can be used for the finest chef cuts but also it can cut yourself or destroy the vegetables if used not properly.

Sam Altman acknowledged there have been issues with the roll out of GPT-5. Matthew Berman tested GPT-5 with a week in advance to its release. He highlighted its capabilities and potential.
