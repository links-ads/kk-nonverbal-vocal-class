# Resources directory

Keep this folder for everything that is neither code nor data, e.g. a file configuration, pretrained weights, or similar.

You can decide whether to push its contents or not.
As rule of thumb, add files to the ignore list if:

 - they contain sensitive material (urls, authentication tokens, environment variables, personal stuff)
 - extremely large files (large model weights, large CSV files)
 - they are not strictly necessary to work with the rest of the repo

On the other hand, push its contents if:

 - they contain useful configuration options or necessary additional files (e.g. a json with class mappings)
 - they are required by the code, and they are not etremely large
 - they are large, but they won't change in time and you're willing to use LFS to push them
