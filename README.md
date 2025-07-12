1. Pull dataset down from AWS, load to array, then dataset then train
    - rename base URL from S3 to current URL
    - make sure use list comprehension to process every item in the array. 
        - dataset.map() will convert image to byte array dict but we need it in PIL Image format for TRL SFTTrainer 