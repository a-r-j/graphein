# Introduction

Yo, what up! if you're reading this then I'm super psyched because that means that you're thinking about contributing to Poly! Thanks so much for your time and consideration. It's rad people like you that make Poly such a cool computational synthetic biology tool.

I wrote this contributor's guide to help newcomers feel welcome. Getting started with a new project can be complicated and I wanted to make it as easy as possible for you to contribute and as easy as possible for me to help.

Currently any sincere pull request is a good request.
Poly is still in [pre-release](https://twitter.com/TimothyStiles/status/1365545416876417028) so there are so many way to contribute!
Here's a list of ideas but feel free to suggest anything I may have forgotten to include.

* Feature requests - especially cool new algorithms with citations.
* Devops features. Github action bots, linters, deployment, etc.
* Unit and integration tests.
* Writing, editing, and translating tutorials, documentation, or blog posts.
* Auditing for accessibility.
* Bug reports.
* Bug triaging.
* Community management.
* Art! Dreams! Your excellence!
* Code that can be pulled into Poly itself.

# Contributor guidelines
### Excellence, and the contributor's code of conduct

First up, most importantly we have a contributor's code of conduct. For some reason the internet is a dehumanizing experience and it's easy to forget that aside from the bots we're all humans on this thing. Approach each other with kindness. Please read our [contributor's code of conduct](CODE_OF_CONDUCT.md) and when in doubt just remember our one true rule as once spoken by the ever so wise duo of Bill and Ted.

`Bill: Be excellent, ... to each other, ...`

`Ted: and party on, dudes! [sic]`

![Abraham Licoln saying, "Be Excellent to each other and party on dudes!". [sic]](https://media.giphy.com/media/ef0zYcF7AKu4b0Sns6/giphy-downsized-large.gif)

### Do-ocracy

Poly runs on do-ocracy. Do-ocracy is a simple concept. If you don't like something you don't need permission to fix it, you can just go ahead and fix it! If you actually want to merge your fix, or contribute in someway that benefits everybody, it'd really, really, really help if you got some light consensus from the rest of the Poly development community but hey, if you really need to do something then you just gotta do it! Just don't expect me to merge it if it doesn't meet our technical criteria or isn't quite right for Poly.

### Technical requirements

Part of what makes Poly so special is that we have standards. DNA is already spaghetti code on its own and we just don't need to add to that.

All successfully merged pull requests must meet the following criteria: 

* All current tests must pass.
 
* At least one new test must be written to prove that the merged feature works correctly.

* At least one new [example test](https://blog.golang.org/examples) must be written to demonstrate the merged feature in our docs.
  
* Build tests must pass for all currently supported systems and package managers. Linux, Mac OSX, Windows, etc.
  
* Code must be clean, readable, and commented. How you do that is up to you!

Don't worry if you submit a pull request and all the tests break and the code is not readable. We won't merge it just yet and then you can get some feedback about what needs to be changed before we do!

### Be welcoming

As one final guideline please be welcoming to newcomers and encourage new contributors from all walks of life. I want Poly to be for everyone and that includes you and people who don't look, sound, or act like you!

# Your first contribution

Unsure where to begin contributing to Poly? You can start by looking through these beginner and help-wanted issues:

[Beginner issues](https://github.com/TimothyStiles/poly/issues?q=is%3Aissue+is%3Aopen+label%3A%22beginner%22+) - issues which should only require a few lines of code, and a test or two.

[Good first issues](https://github.com/TimothyStiles/poly/contribute) - issues which are good for first time contributors.

[Help wanted issues](https://github.com/TimothyStiles/poly/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22+) - issues which should be a bit more involved than beginner issues.

[Feature requests](https://github.com/TimothyStiles/poly/labels/enhancement) - before requesting a new feature search through previous feature requests to see if it's already been requested. If not then feel free to submit a request and tag it with the enhancement tag!

### Working on your first Pull Request? 

You can learn how from this *free* series, [How to Contribute to an Open Source Project on GitHub](https://egghead.io/series/how-to-contribute-to-an-open-source-project-on-github).

You can also check out [these](http://makeapullrequest.com/) [tutorials](http://www.firsttimersonly.com/).

At this point, you're ready to make your changes! Feel free to ask for help; everyone is a beginner at first :smile_cat:

# Getting started

For something that is bigger than a one or two line fix:

1. Create your own fork of the code.
2. Make a branch in your fork
3. Do the changes in your fork's branch.
4. Send a pull request.

## Virtual Environments and Development Containers 

In order to simplify the development experience, and environment setup, the poly Github repository comes packaged to support *Github CodeSpaces* and [*VSCode Development Containers*](https://code.visualstudio.com/docs/remote/containers#_getting-started). *Github CodeSpaces* will give you ability to spin up a Github hosted instance of Ubuntu that would allow you run, test, develop poly from the browser. *VSCode Development Containers* in turn will allow your installation of VSCode to spin up a docker instance of Ubuntu on your computer and automatically mount your code onto it so that you continue developing on this docker instance that has all the required development environment setup. 

## Recommended Plugins

Whether you're a beginner with Go or you're an experienced developer, You should see the suggestions popup automatically when you goto the *Plugins* tab in VSCode. Using these plugins can help accelerate the development experience and also allow you to work more collaboratively with other poly developers.

# How to report a bug

### Security disclosures

If you find a security vulnerability, do NOT open an issue. I've yet to set up a security email for this so please in the interim DM me on twitter for my email [@timothystiles](https://twitter.com/TimothyStiles).

In order to determine whether you are dealing with a security issue, ask yourself these two questions:

* Can I access something that's not mine, or something I shouldn't have access to?
* Can I disable something for other people?
  
If the answer to either of those two questions are "yes", then you're probably dealing with a security issue. Note that even if you answer "no" to both questions, you may still be dealing with a security issue, so if you're unsure, just DM me [@timothystiles](https://twitter.com/TimothyStiles) for my personal email until I can set up a security related email.

### Non-security related bugs

For non-security bug reports please [submit it using this template!](https://github.com/TimothyStiles/poly/issues/new?assignees=&labels=&template=bug_report.md&title=)

# How to suggest a feature or enhancement

If you want to suggest a feature it's as easy as filling out this [issue template](https://github.com/TimothyStiles/poly/issues/new?assignees=&labels=&template=feature_request.md&title=), but before you do please [check to see if it's already been suggested!](https://github.com/TimothyStiles/poly/labels/enhancement)

# How add a recommended VSCode Plugin

Poly comes with a set of recommended plugins for VSCode. If you have suggestions that will simplify life for the poly dev community, consider doing a pull-request after modifying `.vscode/extensions.json`. 

# In closing

Thanks, for reading and I'm super psyched to see what you'll do with Poly!
