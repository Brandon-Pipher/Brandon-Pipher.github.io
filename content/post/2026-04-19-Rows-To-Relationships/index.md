---
date: 2026-04-19
title: "From Rows to Relationships: Using Relational Data for Predicting"
draft: true
toc: false 
---

## Building the relationship

Most predictive models start with a table: one row per example, one set of features per row, one target to predict.

That framing is useful, but it can be incomplete.

In many real problems the primary signal is not stored inside the row at all. It lives in the relationships between entities. Users rate movies. Customers buy products. People follow other people. Articles cite articles. Devices communicate with devices. In these settings, the data is not just a table of attributes. It is a system of interactions.

This post introduces that idea through a concrete example: using user–movie ratings to predict whether a movie is a comedy. The point is not that genre prediction is the most important problem in movie data, or that this is the most effective prediction algorithm. The point is that it gives a clean demonstration of a larger modeling principle:

> Relationships can be predictive, even when the target label belongs to one entity in the system.

Here, the label belongs to the movie: comedy or not comedy. But the predictive signal comes from the movie’s relationships to users through ratings. We will use a simple matrix factorization approach to turn those relationships into latent features, then use those features to predict the label.

This is a deliberately simple setup. It is not the most advanced possible model. That is exactly why it is useful. It shows the core idea with minimal machinery.

## Data

MovieLens provides a clean example dataset. GroupLens publishes several MovieLens datasets built from explicit user ratings of movies; for example, the 1M dataset contains 1 million ratings from 6,000 users on 4,000 movies, and the “latest” dataset page describes a larger benchmark with about 33 million ratings, 2 million tag applications, and 86,000 movies.

That gives us a useful thought experiment:

> Can we predict a movie’s genre from the pattern of users who rated it and how they rated it?

At first glance, that sounds backward. Genre is content. Ratings are behavior. But behavior is not random. Viewers self-select into movies, and their preferences cluster. Those preference structures create relational signal.

A horror movie and a romantic comedy may have similar release years and comparable average ratings, yet they can be rated by very different sets of users. That user-overlap pattern is structure. Structure is learnable.

## Why row-based modeling is not enough

Suppose we try to predict whether a movie is a comedy using only ordinary movie attributes:

release year
runtime
average rating
number of ratings
maybe title words or tags

That can work to some degree. But it misses a major source of information: who watched the movie, and how they responded to it.

A movie is not just an isolated item with metadata. It sits inside a network of behavior. Some groups of users disproportionately watch and enjoy comedies. Other groups gravitate toward horror, thriller, drama, or documentary. If a movie is rated by a set of users whose behavior strongly resembles known comedy-viewing patterns, that relationship structure itself becomes evidence.

This is the key shift:

In standard tabular modeling, features are properties of the object.
In relational modeling, features can come from the object’s links to other objects.

A movie can therefore be represented not only by what it is, but also by how it is connected.

## Why ordinary tabular thinking breaks down
Suppose we build a conventional supervised dataset where each movie is one row and columns are:

- release year
- average rating
- rating count
- runtime
- tag counts
- maybe TF-IDF over title or synopsis

That can work. But it throws away a large fraction of the available information.

The problem is that a movie is not an isolated object. It sits in a web of relations:

users rate movies
users rate many movies
movies are co-rated by overlapping user groups
ratings have values and timestamps
users induce similarity between movies even when the movies have no obvious shared metadata

In a relational view, the object of interest is not just a movie vector $x_m$. It is a node in a bipartite graph or an entry in a sparse interaction matrix.

That change in viewpoint is the whole story.

![TPC-H Schema](tpch_schema.png)

## What relationship data is

Relationship data describes how entities interact.

In this example, we have two types of entities:
- users
- movies

And one type of relationship:
- a user rated a movie

That means the data is not naturally a single flat table. It is more naturally represented as a matrix or a graph.

### Matrix view

We can build a user–movie rating matrix:
$$ R \n \mathbb{R}$$
n
u
	​

×n
m
	​


where:

n
u
	​

 is the number of users
n
m
	​

 is the number of movies
R
u,m
	​

 is the rating user u gave movie m, if one exists

Most entries are missing, because most users rate only a small fraction of movies.

Graph view

The same data can also be seen as a bipartite graph:

one set of nodes = users
one set of nodes = movies
an edge exists if a user rated a movie

Both views describe the same structure. In this post, we will mostly use the matrix view because it leads directly to TruncatedSVD.