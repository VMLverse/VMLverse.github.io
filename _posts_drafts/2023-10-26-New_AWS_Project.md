---
title:  "My First AWS Project"
category: posts
date: 2023-10-26
excerpt: "About my first new AWS project"
toc: false
toc_label: "Contents"
tags:
  - neural network
  - deep learning
  - foundations
  - basics
  - MNIST
---

## Introduction
After completing ML & DL courses in my Machine Learning masters, I never could judge where I was on my ML journey. Although I came across the ML concepts in course work, I always needed to go back to the basics. I never felt confident enough to start a Production ML project or to participate in a Kaggle competition. Remember there are two paths to become an expert: One, to get the foundations solid and then do a project or two, to learn via doing projects. I took up the first path and that is also why I started this blog. I wanted to relearn the basics so I could know them without looking up, do miniprojects, learn some Ops tools like AWS, kubernetes, terraform etc and then do a fullscale project. However, this path seemed never ending as it was one tutorial after another. It became tiring and uninteresting. 

## New Initiative
 At work, I had mentioned to my Manager about my ML Masters and I had no clue I could get an ML project at office since we were doing PS consulting which was a far far stone from anything close to SWE. However, there was this custom solutions team who develop customization on AWS for our clients. Now the Director of this team was tasked with doing anything with Generative AI that would be productive for the PS team. I was called for this new initiative as I was familiar with ML. I was given access to AWS Bedrock (ML Foundation Models for Generative AI) and need to come up with some productive output. Although I was escastatic for this new assignment, my emotion was shortlived. My SWE skills are pretty basic. I haven't used AWS and dont exactly know how I can get over the technical hurdles. Above all, I need to come up with a goal to work towards. Since the world of Generative AI was wide open, I didnt know what I would working towards.

## Brain Storming Goals
I knew I needed to have an output goal or vision  to work towards. Otherwise, changing goals would be a mere learning experience with no productive output. I ran through a couple of use cases to determine a productive goal. 
- **Use Case #1: The ChatBot**
    - The problem: Most of Genesys documents are available on the public internet. There is some collective wisdom among experience folks which is not documented anywhere. PS folks spend time trying to hunt for information.
    - The solution: Create a chatbot for my PS colleagues so they can ask queries and get sensible responses from the chatbot with a link to the article. Integrate this chatbot into the existing group chat for the PS team so team members could provide feedback on answers from the chatbot. The solution could also allow for online training where it monitors the chat and train itself on new information that is chatted among the group members.
    - Why this solution might work: Chatbot being a quick source for information. Information exchanged in Chat groups is being stored.
    - Why this solution might fail: This usecase can already be addressed by an existing Genesys product called Agent Assist. Articles can be loaded into a Knowledge Base  and the chatbot is available as an API. There is also vote up or down options to collect feedback from end user. Moreover, there is not much time being saved through the chatbot. Most of the information is already well documented on the public site and it only takes a new employee a few hours to become familiar with the layout of information.
- **Use Case #2: SoW AI**
    - The problem: Statement of Work  (SoW) are documents for customers that are written by the PS team. The document is customized from a template for each customer. PS effort is spent on customizing this document.
    - The solution: Feed SoWs into a foundational Generative Text AI model to finetune it. Start generating SoWs based on a list of predefined customer requirement fields.
    - why this solution might work: Saves time on SoW manual work.
    - why this solution might fail: Not much effort savings. The SoW document is already available as a template. The PS team already does a find and replace for most fields.  
- **Use Case #2: Code Generation AI**
    - The problem: PS team spends considerable time on building applications in Genesys Architect IVR Application builder. The build takes lot more man hours than Genesys's other IVR building tools like Composer or Designer. Moreover, considerable time is spent in Dev & QA testing. Not to  mention the multitude of human errors possible when building a large application. Moreover some of the applications are hard to build in Architect. PS engineer would know to describe their solution in words, but without extensive Architect experience, it would be difficult to implement it as a callflow.
    - The solution: Create a Interative Code Generation Tool. Honestly, I was inspired from the DALL-E project that takes in descriptive text and outputs image (precursor to MidJourney). A PS Engineer could type in their solution like "build me an input block with 4 options that goes to queue routing" and the callflow block could be built accordingly.  The input texts could also be provided maintained within a session so the tool can construct put together consequetive blocks based on information provide by user and build a complex callflow.
    - Why this solution might work: Huge savings on time as it is the largest spend on any PS project. If the solution could also provide the output code in perfectly standardized format and complete all integrations, then we could be looking at a 100% AI generated error free code.
    - Why this solution might fail: It does seem much more complex. There are a number of hurdles to overcome. First one being, how to collect data and train the model. How