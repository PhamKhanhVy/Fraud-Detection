from django.db import models

class Transaction(models.Model):
    transaction_id = models.CharField(max_length=255, null=True)
    batch_id = models.CharField(max_length=255, null=True)
    account_id = models.CharField(max_length=255, null=True)
    subscription_id = models.CharField(max_length=255, null=True)
    customer_id = models.CharField(max_length=255, null=True)
    currency_code = models.CharField(max_length=255, null=True)
    country_code = models.IntegerField(null=True)
    provider_id = models.CharField(max_length=255, null=True)
    product_id = models.CharField(max_length=255, null=True)
    product_category = models.CharField(max_length=255, null=True)
    channel_id = models.CharField(max_length=255, null=True)
    amount = models.FloatField(null=True)
    value = models.IntegerField(null=True)
    transaction_start_time = models.DateTimeField(null=True)
    pricing_strategy = models.IntegerField(null=True)
    fraud_result = models.IntegerField(null=True)
    dataset = models.CharField(max_length=255, null=False)

    def __str__(self):
        return self.transaction_id