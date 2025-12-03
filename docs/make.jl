using GlobalOpt
using Documenter

DocMeta.setdocmeta!(GlobalOpt, :DocTestSetup, :(using GlobalOpt); recursive=true)

makedocs(;
    modules=[GlobalOpt],
    authors="F. Capolupo",
    repo="https://github.com/FraCpl/GlobalOpt.jl/blob/{commit}{path}#{line}",
    sitename="GlobalOpt.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true", canonical="https://FraCpl.github.io/GlobalOpt.jl", edit_link="master", assets=String[]
    ),
    pages=["Home" => "index.md"],
)

deploydocs(; repo="github.com/FraCpl/GlobalOpt.jl", devbranch="master")
