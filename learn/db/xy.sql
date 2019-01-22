SELECT  X.user,X.click,y.cart,X.shop_id FROM   x LEFT JOIN y ON x.user= y.user and X.shop_id=Y.shop_id
UNION 
SELECT  Y.user,x.click,Y.cart,y.shop_id FROM   x right JOIN y ON x.user= y.user and X.shop_id=Y.shop_id