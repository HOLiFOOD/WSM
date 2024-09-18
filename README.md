# BfR Weak Signal Miner

This project builds on the initial work on weak signal detection by Yoon (2012, https://www.sciencedirect.com/science/article/pii/S0957417412006562).
Weak Signal Detection is a text mining technique aimed at identifying terms or concepts within a text corpus that are potentially underrepresented but may become more relevant or salient in the future. 
This makes weak signal detection a valuable co-indicator alongside other tools developed in WP1.

The underlying premise of weak signals is to distinguish emerging concepts (weak signals) from well-known concepts (strong signals) and non-evolving concepts (noise). 
This is achieved by focusing on concepts that are infrequently mentioned yet exhibit significant changes over time.

**AI Enhancements to Weak Signal Detection**
This repo aims at realizing three AI enhancements in order to improve accuracy and usability of the weak signal approach:
* Turning Term Frequency into Topic Frequency 
* Context Recognition
* LLM based inference of weak signals

## Tasks and tools
* Scrape Abstracts -> get_abstracts.ipynb
* AI enhanced weak signal mining -> WSM.ipynb
* LLM interpreted Weak Signals -> wsm_interpreter.ipynb

## To Do
* make different time units digestable

