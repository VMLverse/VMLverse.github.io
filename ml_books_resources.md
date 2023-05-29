<div align="right" style="font-size: 20px;">
<a href="https://vimvenu-rgb.github.io/">About Me</a> | 
 <a href="blog_contents.html">Blog Contents</a> | 
 <a href="https://vimvenu-rgb.github.io/resume.pdf">Resume</a> | 
 <a href="mailto:vimalkumar.engr@gmail.com?subject=Saw%20Your%20Github%20Blog&body=Enter%20Your%20Text.">Email me</a>
</div>

# GitHub Commands
## Basics
### git commit:
 -   commit in a git repository records a snapshot of all the (tracked) files in your directory
 -   like a giant copy and paste, but even better!
 -   commits as lightweight as possible though, so it doesn't just blindly copy the entire directory every time you commit
<img src="/images/github_commands/git_commit.gif" alt="Alt text" width="512" height="434" />

### git branch:
- simply pointers to a specific commit -- nothing more. branch early, and branch often
<img src="/images/github_commands/git_branch.gif" alt="Alt text" width="512" height="434" />

### git checkout <name> :
- put us on the new branch before committing our changes

<img src="/images/github_commands/git_checkout.gif" alt="Alt text" width="512" height="434" />
 
### git checkout -b [yourbranchname]:
- create a new branch AND check it out at the same time

<img src="/images/github_commands/git_checkout_b.gif" alt="Alt text" width="512" height="434" />
 
### git merge:
- Merging in Git creates a special commit that has two unique parents
<img src="/images/github_commands/git_merge.gif" alt="Alt text" width="512" height="434" />

### git rebase:
- Rebasing essentially takes a set of commits, "copies" them, and plops them down somewhere else
<img src="/images/github_commands/git_rebase.gif" alt="Alt text" width="512" height="434" />

## HEAD
- HEAD is the symbolic name for the currently checked out commit 
- it's essentially what commit you're working on top of.
- HEAD always points to the most recent commit which is reflected in the working tree.
 
### detach head:
- Detaching HEAD just means attaching it to a commit instead of a branch.
- To detach simply specify checkout by its hash

<img src="/images/github_commands/git_detach.gif" alt="Alt text" width="512" height="434" />
 
### Relative commits:
#### Caret (^) operator:
- Moving upwards one commit at a time with ^
- Each time you append that to a ref name, you are telling Git to find the parent of the specified commit.
- saying main^ is equivalent to "the first parent of main"
<img src="/images/github_commands/git_caret.gif" alt="Alt text" width="512" height="434" />
#### tilde (~) operator:
- Moving upwards a number of times with ~<num>
- tilde operator (optionally) takes in a trailing number that specifies the number of parents you would like to ascend
<img src="/images/github_commands/git_tilde.gif" alt="Alt text" width="512" height="434" />

#### branch by force(-f):
- You can directly reassign a branch to a commit with the -f option
<img src="/images/github_commands/git_branch_f.gif" alt="Alt text" width="512" height="434" />
 
### Reversing Changes:
#### git reset:
- reverses changes by moving a branch reference backwards in time to an older commit
<img src="/images/github_commands/git_reset.gif" alt="Alt text" width="512" height="434" />

#### git revert:
- git reset doesn't work for remote branches that others are using.
- In order to reverse changes and share those reversed changes with others, we need to use git revert.
<img src="/images/github_commands/git_revert.gif" alt="Alt text" width="512" height="434" />
<!-- [add immage] ![Alt text](/images/positive_negative.png) -->
<!-- [heading] ## Precision & Recall -->
<!-- [subheading] ### Precision -->

## References
- Learn git branching: https://learngitbranching.js.org/
