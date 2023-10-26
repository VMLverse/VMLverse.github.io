---
title:  "Markdown Cheatsheet: Quick Reference for Commonly Used Syntax"
category: posts
date: 2023-06-08
excerpt: "This cheatsheet provides a concise reference for frequently used Markdown formatting, helping you create beautiful, well-structured content with ease."
toc: true
toc_label: "Contents"
tags:
  - markdown
  - jekyll
---


## The Basics

### Newline
You can create a newline by adding two spaces at the end of a line.  
```
This is the first line.  
This is the second line.
```
This is the first line.  
This is the second line.

### New paragraph
You can create a new paragraph by leaving a blank line between the lines of text. 
```
This is the last line.

This is a new paragraph.
```
This is the last line.

This is a new paragraph.

### Bold & Italic
You can italicize text by using single asterisks ('*') or single underscores (_).  
You can make text bold by using double asterisks (**) or double underscores (__)  
You can make text bold & italic by using triple asterisks (***) or triple underscores (___)  
```
This is *italic* text.
This is _also_ italic text.

This is **bold** text.
This is __also__ bold text.

This is ***bold and italic*** text.
This is ___also bold and italic___ text.
```
This is *italic* text.  
This is _also_ italic text.

This is **bold** text.  
This is __also__ bold text.

This is ***bold and italic*** text.  
This is ___also bold and italic___ text.

### Escape characters
You can use a backslash (\) to escape characters that have special meaning and prevent them from being interpreted as formatting or syntax.  
```
To display a literal asterisk: \*
To display a literal underscore: \_
To display a literal backslash: \\
To display a literal square bracket: \[ or \]
To display a literal hash symbol: \#
```
To display a literal asterisk: \*  
To display a literal underscore: \_  
To display a literal backslash: \\  
To display a literal square bracket: \[ or \]  
To display a literal hash symbol: \#  

### Links

1. Inline Links:
   - Syntax: `[link text](url)`
   - Example: `[Visit our website](https://www.example.com)`
   - Renders as: [Visit our website](https://www.google.com)

2. Inline Links with Title Attribute:
   - Syntax: `[link text](url "title")`
   - Example: `[Open in new tab](https://www.example.com "Opens in a new tab")`
   - Renders as: [Open in new tab](https://www.example.com "Opens in a new tab Title")
   - `Opens in a new tab Title` is shown when hovering over the link.

3. Reference-style Links:
   - Syntax: `[link text][reference]` and `[reference]: url "title"`
   - Example:
     ```
     [Learn more][example]
     [example]: https://www.example.com "Visit our website"
     ```

4. Inline Image Links:
   - Syntax: `![alt text](image url)`
   - Example: `![Logo](https://www.example.com/images/logo.png)`

5. Inline Image Links with Title Attribute:
   - Syntax: `![alt text](image url "title")`
   - Example: `![Logo](https://www.example.com/images/logo.png "Company Logo")`

6. Reference-style Image Links:
   - Syntax: `![alt text][reference]` and `[reference]: image url "title"`
   - Example:
     ```
     ![Logo][logo]
     [logo]: https://www.example.com/images/logo.png "Company Logo"
     ```

Remember to replace the `link text`, `url`, `alt text`, and `image url` with your desired values. The `title` attribute is optional and can be used to provide additional information when the user hovers over the link or image.

These Markdown link syntaxes allow you to create various types of links within your Markdown content, enabling seamless navigation and enhancing user experience. I personally found the reference-style links to be useful since you can have a footer with all your referenced URLs in your article and maintain them in one place. 

### Images
To add images in Markdown, you can use the following syntax:

1. Inline Image:
   - Syntax: `![alt text](image URL)`
   - Example: `![Example Image](https://www.example.com/images/example.jpg)`

2. Reference-style Image:
   - Syntax: `![alt text][reference]` and `[reference]: image URL`
   - Example:
     ```
     ![Example Image][image-ref]
     [image-ref]: https://www.example.com/images/example.jpg
     ```

In Markdown, the basic syntax does not provide direct control over image size or position. However, you can use HTML attributes within Markdown to achieve these effects. Here's how you can size and position images using HTML attributes:

1. Sizing Images:
   - Use the `width` attribute to specify the desired width of the image in pixels or percentage.
   - Use the `height` attribute to specify the desired height of the image in pixels or percentage.
   - Example:
     ```markdown
     <img src="image.jpg" alt="Example Image" width="300" height="200">
     ```

2. Positioning Images:
   - Use CSS styling within an HTML `<div>` or `<figure>` element to control the positioning of the image.
   - Apply CSS properties like `float`, `margin`, or `text-align` to position the image as desired.
   - Example:
     ```markdown
     <div style="float: right; margin: 10px;">
       <img src="image.jpg" alt="Example Image">
     </div>
     ```

3. Combining Sizing and Positioning:
   - You can combine the sizing and positioning techniques to control both the size and position of the image.
   - Example:
     ```markdown
     <div style="float: right; margin: 10px;">
       <img src="image.jpg" alt="Example Image" width="300" height="200">
     </div>
     ```     

### Code Blocks
To write code in Markdown, you can use the following techniques:

1. Inline Code: Use backticks (\`) to enclose the code within a sentence or paragraph. This helps differentiate code snippets from regular text. For example: 

   Markdown code: `` `print("Hello, World!")` ``

   Rendered output: `print("Hello, World!")`

2. Code Blocks: To display a larger block of code, you can use fenced code blocks. Place triple backticks (\```) on a separate line above and below the code. Optionally, specify the programming language for syntax highlighting after the opening backticks. For example:

   Markdown code:
   ````
   ```python
   def greet(name):
       print("Hello, " + name + "!")
   ```
   ````

   Rendered output:
   ```python
   def greet(name):
       print("Hello, " + name + "!")
   ```

   Note: Some Markdown processors may require a blank line before and after the fenced code block.

3. Syntax Highlighting: To enable syntax highlighting for the code blocks, specify the programming language after the opening backticks. This will apply appropriate color coding to improve code readability. For example:

   Markdown code:
   ````
   ```javascript
   function add(a, b) {
       return a + b;
   }
   ```
   ````

   Rendered output:
   ```javascript
   function add(a, b) {
       return a + b;
   }
   ```

Commonly supported programming languages for syntax highlighting include JavaScript, Python, Java, C++, HTML, CSS, and more.

## Buttons

Make any link standout more when applying the `.btn .btn--primary` classes.

```html
<a href="#" class="btn btn--primary">Link Text</a>
```

| Button Type   | Example | Class | Kramdown |
| ------        | ------- | ----- | ------- |
| Default       | [Text](#link){: .btn} | `.btn` | `[Text](#link){: .btn}` |
| Primary       | [Text](#link){: .btn .btn--primary} | `.btn .btn--primary` | `[Text](#link){: .btn .btn--primary}` |
| Success       | [Text](#link){: .btn .btn--success} | `.btn .btn--success` | `[Text](#link){: .btn .btn--success}` |
| Warning       | [Text](#link){: .btn .btn--warning} | `.btn .btn--warning` | `[Text](#link){: .btn .btn--warning}` |
| Danger        | [Text](#link){: .btn .btn--danger} | `.btn .btn--danger` | `[Text](#link){: .btn .btn--danger}` |
| Info          | [Text](#link){: .btn .btn--info} | `.btn .btn--info` | `[Text](#link){: .btn .btn--info}` |
| Inverse       | [Text](#link){: .btn .btn--inverse} | `.btn .btn--inverse` | `[Text](#link){: .btn .btn--inverse}` |
| Light Outline | [Text](#link){: .btn .btn--light-outline} | `.btn .btn--light-outline` | `[Text](#link){: .btn .btn--light-outline}` |

| Button Size | Example | Class | Kramdown |
| ----------- | ------- | ----- | -------- |
| X-Large     | [X-Large Button](#){: .btn .btn--primary .btn--x-large} | `.btn .btn--primary .btn--x-large` | `[Text](#link){: .btn .btn--primary .btn--x-large}` |
| Large       | [Large Button](#){: .btn .btn--primary .btn--large} | `.btn .btn--primary .btn--large` | `[Text](#link){: .btn .btn--primary .btn--large}` |
| Default     | [Default Button](#){: .btn .btn--primary} | `.btn .btn--primary` | `[Text](#link){: .btn .btn--primary }` |
| Small       | [Small Button](#){: .btn .btn--primary .btn--small} | `.btn .btn--primary .btn--small` | `[Text](#link){: .btn .btn--primary .btn--small}` |

## Notices

Call attention to a block of text.

| Notice Type | Class              |
| ----------- | -----              |
| Default     | `.notice`          |
| Primary     | `.notice--primary` |
| Info        | `.notice--info`    |
| Warning     | `.notice--warning` |
| Success     | `.notice--success` |
| Danger      | `.notice--danger`  |

**Watch out!** This paragraph of text has been [emphasized](#) with the `{: .notice}` class.
{: .notice}

**Watch out!** This paragraph of text has been [emphasized](#) with the `{: .notice--primary}` class.
{: .notice--primary}

**Watch out!** This paragraph of text has been [emphasized](#) with the `{: .notice--info}` class.
{: .notice--info}

**Watch out!** This paragraph of text has been [emphasized](#) with the `{: .notice--warning}` class.
{: .notice--warning}

**Watch out!** This paragraph of text has been [emphasized](#) with the `{: .notice--success}` class.
{: .notice--success}

**Watch out!** This paragraph of text has been [emphasized](#) with the `{: .notice--danger}` class.
{: .notice--danger}

{% capture notice-text %}
You can also add the `.notice` class to a `<div>` element.

* Bullet point 1
* Bullet point 2
{% endcapture %}

<div class="notice--info">
  <h4 class="no_toc">Notice Headline:</h4>
  {{ notice-text | markdownify }}
</div>


## References
* [Jetbrains Markdown Syntax](https://www.jetbrains.com/help/hub/Markdown-Syntax.html)
* [Markdown Guide](https://www.markdownguide.org/getting-started/)
