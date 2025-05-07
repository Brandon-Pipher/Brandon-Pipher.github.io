# [Hugo Academic CV Theme](https://github.com/HugoBlox/theme-academic-cv)
https://gohugo.io/

https://hugoblox.com/

# Set up
Steps are more specifically tailored to windows and are easier on linux. I leverage git bash.

Clone theme repository (theme-academic-cv from hugoblox).  Github Desktop can always be used.

Install Git, Go and Hugo (easy using winget).  All of this can be ran from VS Code Powershell Terminal (not Windows Powershell).

Use 'hugo' to set up and 'hugo server -D' to serve locally with draft posts.

Pull ./.github/workflows/import-notebooks.yml from another repo and create ./notebooks/ directory. Allows ipynb conversions.

Needs a .gitignore to be created.