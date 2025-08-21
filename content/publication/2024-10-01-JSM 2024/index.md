---
title: 'Analyzing the Residential Status of Demographic Frame Addresses'

# Authors
# If you created a profile for a user (e.g. the default `admin` user), write the username (folder name) here
# and it will be replaced with their full name and linked to their profile.
authors:
  - admin
  - Tom Mule

# Author notes (optional)
author_notes:

date: '2024-10-01'
#doi: '10.5281/zenodo.14009748'

# Schedule page publish date (NOT publication's date).
publishDate: '2024-10-01'

# Publication type.
# Accepts a single type but formatted as a YAML list (for Hugo requirements).
# Enter a publication type from the CSL standard.
publication_types: ['paper-conference']

# Publication name and optional abbreviated publication name.
publication: ASA Joint Statistical Meetings
publication_short: JSM

abstract: The Census Bureau’s Demographic Frame is a comprehensive, person-level frame consisting of geographic, demographic, social, and economic characteristics. It could operate as a sampling frame for household surveys, reducing respondent burden by using information already available to the federal government, or improving data quality by drawing from the frame for data editing and imputation. It can be used to identify addresses associated with each person and potentially to identify their residence. Person-address records on the Demographic Frame are derived from administrative, third-party, census and survey data records. A person found on the Demographic Frame may have multiple address records, as they are often associated with several addresses across the various data sources. These multiple address-records create difficulty in placing a person at their correct residential address according to a given reference day. The Demographic Frame has a person-place model process that assigns probabilities to each person-address record and can be used to determine a person's residence on a particular reference date. These models learn from person-place pairs on existing data within the Census Bureau to make predictions about other person-place pairs derived from administrative records. This analysis will evaluate the accuracy of addresses for people from these models on the Demographic Frame based on a reference date of July 1, 2021. As part of this analysis, we will compare these Demographic Frame addresses to addresses in the 2020 Census and the 2021 American Community Survey frames. This comparison will allow us to examine the Demographic Frame addresses that are not found within these other Census products, which may provide information to help identify whether subsets of these addresses may be more likely to be residential or non-residential addresses. This analysis can potentially improve the quality of the Demographic Frame and its person-place records by increasing the chance of placing a person at their correct residential address.

# Summary. An optional shortened abstract.
summary: Analysis on the Residential Status of living quarters found on the Census Bureau's Demographic Frame 

tags:

# Display this page in the Featured widget?
featured: true

# Custom links (uncomment lines below)
# links:
# - name: Custom Link
#   url: http://example.org

url_pdf: ''
url_code: ''
url_dataset: ''
url_poster: ''
url_project: ''
url_slides: ''
url_source: ''
url_video: ''

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
image:
  caption: ''
  focal_point: 'Center'
  preview_only: false

# Associated Projects (optional).
#   Associate this publication with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `internal-project` references `content/project/internal-project/index.md`.
#   Otherwise, set `projects: []`.
projects:

# Slides (optional).
#   Associate this publication with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides: "example"` references `content/slides/example/index.md`.
#   Otherwise, set `slides: ""`.
slides:
draft: false
---
