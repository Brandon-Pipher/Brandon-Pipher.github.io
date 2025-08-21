# [Hugo Academic CV Theme](https://github.com/HugoBlox/theme-academic-cv)
https://gohugo.io/

https://hugoblox.com/

# Set up
Steps are more specifically tailored to windows and are easier on linux. I leverage git bash for windows.

Install Git, Go and Hugo (easy using winget).  All of this can be ran from VS Code Powershell Terminal (not Windows Powershell).

Remember to verify that ~/.bashrc has the following if using windows git bash to work from:
    source "C:\Program Files\Git\mingw64\share\git\completion\git-prompt.sh"
    source "C:\Users\bppipher\anaconda3\etc\profile.d\conda.sh"

Clone theme repository (theme-academic-cv from hugoblox).

Use 'hugo' to set up and 'hugo server -D' to serve locally with draft posts. Hugo server needs to be correct to work. Check version built with in hugoblox.yaml

Pull ./.github/workflows/import-notebooks.yml from another repo and create ./notebooks/ directory. Allows ipynb conversions.

Needs a .gitignore to be created.


# Getting Started on ubuntu

Verify the version of `hugo` needed by checking hugoblox.yaml (0.136.5-extended)

Can be installed using the following:

```
wget https://github.com/gohugoio/hugo/releases/download/v0.136.5/hugo_extended_0.136.5_linux-amd64.deb
sudo dpkg -i hugo_extended_0.136.5_linux-amd64.deb || sudo apt -f install
```

Will also need `go` and `nodejs` and is easily installed using snap:

```
sudo snap install --classic go
sudo snap install --classic node
```